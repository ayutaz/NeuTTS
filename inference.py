#!/usr/bin/env python3
"""
inference.py - MSpoof-TTS の推論エントリポイント

階層的スプーフ誘導デコーディングを用いて高品質な音声合成を行う。
ベースTTS（NeuTTS）は凍結状態のまま、訓練済みスプーフ検出器を利用して
デコーディング過程でトークン列の品質を評価・制御する。

使い方 (EASモード):
    python inference.py \
        --input-text "Hello, this is a test." \
        --reference-audio ref.wav \
        --output output.wav \
        --mode eas

使い方 (階層的モード / Phase 5):
    python inference.py \
        --input-text "Hello, this is a test." \
        --reference-audio ref.wav \
        --output output.wav \
        --mode hierarchical \
        --checkpoint-dir ./checkpoints/detectors
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from mspoof_tts.config import (
    EASConfig,
    HierarchicalConfig,
    InferenceConfig,
    load_inference_config,
)
from mspoof_tts.sampling.eas import EntropyAwareSampler
from neutts import NeuTTS

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# カスタム生成ループ: EASを用いた自己回帰トークン生成
# ---------------------------------------------------------------------------

def generate_with_eas(
    model: NeuTTS,
    prompt_ids: list[int],
    eas_sampler: EntropyAwareSampler,
    max_new_tokens: int = 2048,
    device: str = "cpu",
) -> list[int]:
    """EASを用いたカスタム自己回帰生成ループ。

    NeuTTS の backbone (HuggingFace CausalLM) の forward() を直接呼び出し、
    各ステップでEASサンプリングを適用してトークンを生成する。

    Args:
        model: NeuTTS インスタンス。backbone と tokenizer にアクセスする。
        prompt_ids: プロンプトのトークンID列 (list[int])。
        eas_sampler: EntropyAwareSampler インスタンス。
        max_new_tokens: 生成する最大トークン数。
        device: 推論デバイス ("cpu" / "cuda" / "mps" など)。

    Returns:
        生成されたトークンID列 (プロンプトを含まない)。
    """
    backbone = model.backbone
    tokenizer = model.tokenizer

    eos_token_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    # サンプラーの状態をリセット
    eas_sampler.reset()

    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
    generated_ids: list[int] = []

    # KV キャッシュの利用で推論を高速化
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                # KVキャッシュがある場合は最後のトークンのみを入力
                outputs = backbone(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # 最初のステップはプロンプト全体を入力
                outputs = backbone(
                    input_ids,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            past_key_values = outputs.past_key_values

        # EASサンプリング: logitsは1D (vocab_size,) で渡す
        next_token = eas_sampler.sample_step(logits.squeeze(0))

        next_token_id = next_token.item()

        # EOS チェック
        if next_token_id == eos_token_id:
            logger.info("EOS トークンを検出しました (ステップ %d)", step + 1)
            break

        generated_ids.append(next_token_id)

        # 次のステップ用に入力を更新 (KVキャッシュ使用時は最後のトークンだけ追加)
        input_ids = torch.cat(
            [input_ids, next_token.view(1, 1).to(device)], dim=-1
        )

        # 進捗ログ (100トークンごと)
        if (step + 1) % 100 == 0:
            logger.info("  生成中... %d トークン生成済み", step + 1)

    if len(generated_ids) == max_new_tokens:
        logger.warning(
            "最大トークン数 (%d) に到達しました。生成を打ち切ります。", max_new_tokens
        )

    return generated_ids


def generate_with_hierarchical(
    model: NeuTTS,
    prompt_ids: list[int],
    hierarchical_decoder: "HierarchicalDecoder",
    max_new_tokens: int = 2048,
    device: str = "cpu",
) -> list[int]:
    """階層的デコーディングによるトークン生成。

    HierarchicalDecoder.generate() を使用して、ウォームアップ後に
    3ステージのビームサーチ・枝刈り・ランク集約を反復的に実行する。

    Args:
        model: NeuTTS インスタンス。backbone と tokenizer にアクセスする。
        prompt_ids: プロンプトのトークンID列 (list[int])。
        hierarchical_decoder: HierarchicalDecoder インスタンス。
        max_new_tokens: 生成する最大トークン数。
        device: 推論デバイス ("cpu" / "cuda" / "mps" など)。

    Returns:
        生成されたトークンID列 (プロンプトを含まない)。
    """
    backbone = model.backbone
    tokenizer = model.tokenizer

    eos_token_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    prefix = torch.tensor(prompt_ids, dtype=torch.long).to(device)

    # HierarchicalDecoder.generate() は backbone を直接使用する
    # speaker_tokens は既にプロンプトに含まれているため空テンソルを渡す
    generated = hierarchical_decoder.generate(
        model=backbone,
        text_tokens=prefix,
        speaker_tokens=torch.tensor([], dtype=torch.long).to(device),
    )

    return generated.tolist()


def tokens_to_speech_string(
    model: NeuTTS,
    generated_ids: list[int],
) -> str:
    """生成されたトークンIDを <|speech_N|> 形式の文字列に変換する。

    NeuTTS の _decode() は "<|speech_123|><|speech_456|>..." 形式の文字列を
    期待するため、生成されたトークンIDをデコードしてこの形式に変換する。

    Args:
        model: NeuTTS インスタンス。
        generated_ids: 生成されたトークンID列。

    Returns:
        "<|speech_N|>" トークンのみを含む文字列。
    """
    tokenizer = model.tokenizer
    output_str = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return output_str


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="MSpoof-TTS 階層的スプーフ誘導デコーディングによる音声合成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------
    # 入出力の設定
    # -------------------------------------------------------------------
    io_group = parser.add_argument_group("入出力の設定")

    io_group.add_argument(
        "--input-text",
        type=str,
        default=None,
        help="合成するテキスト（直接指定）",
    )
    io_group.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="入力テキストファイルのパス（1行に1文）",
    )
    io_group.add_argument(
        "--reference-audio",
        type=str,
        required=True,
        help="リファレンス音声ファイルのパス（話者特性の条件付けに使用）",
    )
    io_group.add_argument(
        "--reference-text",
        type=str,
        default=None,
        help="リファレンス音声のテキスト（未指定の場合、.txt ファイルから自動読み込みを試行）",
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="./output.wav",
        help="出力WAVファイルのパス",
    )
    io_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="訓練済みスプーフ検出器のチェックポイントディレクトリ（hierarchicalモード時に必須）",
    )

    # -------------------------------------------------------------------
    # デコーディングモードの設定
    # -------------------------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eas", "hierarchical"],
        default="eas",
        help="デコーディングモード: 'eas'（EASのみ）または 'hierarchical'（EAS+階層的ビームサーチ, Phase 5）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="推論設定YAMLファイルのパス（未指定の場合はデフォルト設定を使用）",
    )

    # -------------------------------------------------------------------
    # Entropy-Aware Sampling (EAS) のパラメータ
    # -------------------------------------------------------------------
    eas_group = parser.add_argument_group(
        "Entropy-Aware Sampling (EAS) のパラメータ"
    )

    eas_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top-k サンプリングの k 値",
    )
    eas_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="nucleus sampling の累積確率閾値 p",
    )
    eas_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="サンプリング温度",
    )
    eas_group.add_argument(
        "--eas-alpha",
        type=float,
        default=None,
        help="繰り返しペナルティの強度 alpha",
    )
    eas_group.add_argument(
        "--eas-beta",
        type=float,
        default=None,
        help="時間減衰率 beta",
    )
    eas_group.add_argument(
        "--eas-gamma",
        type=float,
        default=None,
        help="Nucleus sampling の閾値 gamma",
    )
    eas_group.add_argument(
        "--cluster-size",
        type=int,
        default=None,
        help="エントロピー閾値パラメータ k_e",
    )
    eas_group.add_argument(
        "--memory-window",
        type=int,
        default=None,
        help="メモリバッファのウィンドウサイズ W",
    )

    # -------------------------------------------------------------------
    # 階層的ビームサーチのパラメータ
    # -------------------------------------------------------------------
    hier_group = parser.add_argument_group("階層的ビームサーチのパラメータ")

    hier_group.add_argument(
        "--warmup-length",
        type=int,
        default=None,
        help="ウォームアップフェーズの長さ L_w（EAS のみで生成する初期トークン数）",
    )
    hier_group.add_argument(
        "--beam-sizes",
        type=int,
        nargs=3,
        default=None,
        help="各ステージのビーム幅 (B0, B1, B2)",
    )
    hier_group.add_argument(
        "--stage-lengths",
        type=int,
        nargs=3,
        default=None,
        help="各ステージのセグメント長 (L1, L2, L3)",
    )
    hier_group.add_argument(
        "--rank-weights",
        type=float,
        nargs=3,
        default=None,
        help="最終ランク集約の重み (w_50, w_25, w_10)",
    )

    # -------------------------------------------------------------------
    # その他の設定
    # -------------------------------------------------------------------
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="生成する最大トークン数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（再現性のため）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推論デバイス (cpu / cuda / mps)。未指定の場合は自動検出。",
    )
    parser.add_argument(
        "--backbone-repo",
        type=str,
        default="neuphonic/neutts-nano",
        help="NeuTTS backbone モデルのリポジトリ名またはパス",
    )
    parser.add_argument(
        "--codec-repo",
        type=str,
        default="neuphonic/neucodec",
        help="NeuCodec のリポジトリ名またはパス",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="半精度（FP16）で推論を行う（GPUメモリ節約のため）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 設定の構築（YAML + CLIオーバーライド）
# ---------------------------------------------------------------------------

def build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    """YAMLファイルとCLI引数から InferenceConfig を構築する。

    YAMLで読み込んだ設定をベースに、CLI引数で明示的に指定された値で上書きする。

    Args:
        args: パースされたコマンドライン引数。

    Returns:
        InferenceConfig オブジェクト。
    """
    # ベース設定をYAMLまたはデフォルトから読み込む
    if args.config is not None:
        config = load_inference_config(Path(args.config))
    else:
        # デフォルト設定ファイルが存在すればそれを使用、なければデフォルト値
        default_config_path = Path(__file__).resolve().parent / "configs" / "inference.yaml"
        if default_config_path.exists():
            config = load_inference_config(default_config_path)
        else:
            config = InferenceConfig()

    # CLI引数でEAS設定を上書き
    if args.top_k is not None:
        config.eas.top_k = args.top_k
    if args.top_p is not None:
        config.eas.top_p = args.top_p
    if args.temperature is not None:
        config.eas.temperature = args.temperature
    if args.cluster_size is not None:
        config.eas.cluster_size = args.cluster_size
    if args.memory_window is not None:
        config.eas.memory_window = args.memory_window
    if args.eas_alpha is not None:
        config.eas.alpha = args.eas_alpha
    if args.eas_beta is not None:
        config.eas.beta = args.eas_beta
    if args.eas_gamma is not None:
        config.eas.gamma = args.eas_gamma

    # CLI引数で階層的デコーディング設定を上書き
    if args.warmup_length is not None:
        config.hierarchical.warmup_length = args.warmup_length
    if args.beam_sizes is not None:
        config.hierarchical.beam_sizes = args.beam_sizes
    if args.stage_lengths is not None:
        config.hierarchical.stage_lengths = args.stage_lengths
    if args.rank_weights is not None:
        config.hierarchical.rank_weights = args.rank_weights

    return config


# ---------------------------------------------------------------------------
# 入力テキストの読み込み
# ---------------------------------------------------------------------------

def load_input_text(args: argparse.Namespace) -> list[str]:
    """入力テキストを読み込む。

    --input-text が指定されていればそのテキストを使用し、
    --input-file が指定されていればファイルから読み込む。
    両方未指定の場合はエラー。

    Args:
        args: パースされたコマンドライン引数。

    Returns:
        合成対象のテキスト行のリスト。
    """
    if args.input_text is not None:
        return [args.input_text.strip()]

    if args.input_file is not None:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(
                f"入力テキストファイルが見つかりません: {args.input_file}"
            )
        lines = input_path.read_text(encoding="utf-8").strip().splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            raise ValueError(f"入力テキストファイルが空です: {args.input_file}")
        return lines

    raise ValueError("--input-text または --input-file のいずれかを指定してください。")


def load_reference_text(args: argparse.Namespace) -> str:
    """リファレンス音声のテキストを取得する。

    --reference-text が指定されていればそれを使用し、
    未指定の場合はリファレンス音声ファイルと同名の .txt ファイルを探す。

    Args:
        args: パースされたコマンドライン引数。

    Returns:
        リファレンスのテキスト文字列。
    """
    if args.reference_text is not None:
        return args.reference_text.strip()

    # リファレンス音声と同名の .txt ファイルを探す
    ref_path = Path(args.reference_audio)
    txt_path = ref_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if text:
            logger.info("リファレンステキストを自動検出しました: %s", txt_path)
            return text

    raise ValueError(
        "リファレンステキストが指定されていません。"
        "--reference-text を指定するか、リファレンス音声と同名の .txt ファイルを配置してください。"
    )


# ---------------------------------------------------------------------------
# デバイスの自動検出
# ---------------------------------------------------------------------------

def resolve_device(device_arg: Optional[str]) -> str:
    """推論デバイスを決定する。

    Args:
        device_arg: CLI引数で指定されたデバイス名。Noneの場合は自動検出。

    Returns:
        デバイス文字列 ("cpu", "cuda", "mps" など)。
    """
    if device_arg is not None:
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 引数の検証
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    """引数の整合性を検証する。"""

    # リファレンス音声ファイルの存在確認
    reference_path = Path(args.reference_audio)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"リファレンス音声ファイルが見つかりません: {args.reference_audio}"
        )

    # hierarchical モードではチェックポイントディレクトリが必要
    if args.mode == "hierarchical":
        if args.checkpoint_dir is None:
            raise ValueError(
                "hierarchical モードでは --checkpoint-dir の指定が必要です。"
            )
        ckpt_path = Path(args.checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"チェックポイントディレクトリが見つかりません: {args.checkpoint_dir}"
            )
        # 5つの検出器チェックポイントの存在確認
        detector_names = ["M_10", "M_25", "M_50", "M_50_25", "M_50_10"]
        for name in detector_names:
            detector_path = ckpt_path / f"{name}.pt"
            if not detector_path.exists():
                raise FileNotFoundError(
                    f"検出器のチェックポイントが見つかりません: {detector_path}"
                )

    # パラメータ値の範囲チェック (CLI引数が明示されている場合のみ)
    if args.top_p is not None and not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"top-p は (0, 1] の範囲で指定してください: {args.top_p}")

    if args.temperature is not None and args.temperature <= 0.0:
        raise ValueError(
            f"temperature は正の値で指定してください: {args.temperature}"
        )

    if args.beam_sizes is not None:
        if len(args.beam_sizes) != 3:
            raise ValueError("beam-sizes は3つの値を指定してください")
        if not (args.beam_sizes[0] >= args.beam_sizes[1] >= args.beam_sizes[2]):
            logger.warning(
                "ビーム幅が段階的に減少していません: %s（通常は B0 >= B1 >= B2）",
                args.beam_sizes,
            )


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    """推論のメインエントリポイント。"""

    args = parse_args()

    # ------------------------------------------------------------------
    # 1. 設定のセットアップ
    # ------------------------------------------------------------------
    logger.info("========================================")
    logger.info("MSpoof-TTS 推論")
    logger.info("========================================")

    # モード情報のログ出力
    if args.mode == "eas":
        logger.info("デコーディングモード: EAS（Entropy-Aware Sampling のみ）")
    elif args.mode == "hierarchical":
        logger.info(
            "デコーディングモード: 階層的（EAS + マルチ解像度スプーフガイド付きビームサーチ）"
        )
        logger.info("  チェックポイントディレクトリ: %s", args.checkpoint_dir)

    # 引数の検証
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error("引数の検証に失敗しました: %s", e)
        sys.exit(1)

    # InferenceConfig の構築 (YAML + CLIオーバーライド)
    config = build_inference_config(args)

    # デバイスの決定
    device = resolve_device(args.device)

    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("")
    logger.info("設定:")
    logger.info("  モード:           %s", args.mode)
    logger.info("  デバイス:         %s", device)
    logger.info("  シード:           %d", args.seed)
    logger.info("  最大トークン数:   %d", args.max_tokens)
    logger.info("  backbone:         %s", args.backbone_repo)
    logger.info("  codec:            %s", args.codec_repo)
    logger.info("")
    logger.info("EAS パラメータ:")
    logger.info(
        "  top-k=%d, top-p=%.2f, temperature=%.2f",
        config.eas.top_k, config.eas.top_p, config.eas.temperature,
    )
    logger.info(
        "  alpha=%.2f, beta=%.2f, gamma=%.2f",
        config.eas.alpha, config.eas.beta, config.eas.gamma,
    )
    logger.info(
        "  cluster_size=%d, memory_window=%d",
        config.eas.cluster_size, config.eas.memory_window,
    )

    if args.mode == "hierarchical":
        logger.info("")
        logger.info("階層的デコーディング パラメータ:")
        logger.info("  warmup_length=%d", config.hierarchical.warmup_length)
        logger.info("  beam_sizes=%s", config.hierarchical.beam_sizes)
        logger.info("  stage_lengths=%s", config.hierarchical.stage_lengths)
        logger.info("  rank_weights=%s", config.hierarchical.rank_weights)
    logger.info("")

    # ------------------------------------------------------------------
    # 2. モデルのロード
    # ------------------------------------------------------------------
    logger.info("NeuTTS モデルをロードしています...")
    tts_model = NeuTTS(
        backbone_repo=args.backbone_repo,
        backbone_device=device,
        codec_repo=args.codec_repo,
        codec_device=device,
    )

    # GGUF (量子化) モデルはカスタム生成ループに対応していない
    if tts_model._is_quantized_model:
        logger.error(
            "EAS モードでは PyTorch backbone が必要です。"
            "GGUF モデルはカスタム生成ループに対応していません。"
            "PyTorch モデル (例: neuphonic/neutts-nano) を指定してください。"
        )
        sys.exit(1)

    # FP16への変換 (GPU使用時)
    if args.use_fp16 and device != "cpu":
        logger.info("backbone を FP16 に変換しています...")
        tts_model.backbone = tts_model.backbone.half()

    # EASサンプラーの作成
    logger.info("EntropyAwareSampler を初期化しています...")
    eas_sampler = EntropyAwareSampler(
        top_k=config.eas.top_k,
        top_p=config.eas.top_p,
        temperature=config.eas.temperature,
        cluster_size=config.eas.cluster_size,
        memory_window=config.eas.memory_window,
        alpha=config.eas.alpha,
        beta=config.eas.beta,
        gamma=config.eas.gamma,
    )

    # 階層的デコーダーの準備 (hierarchical モード時のみ)
    hierarchical_decoder = None
    if args.mode == "hierarchical":
        from mspoof_tts.models.multi_resolution import MultiResolutionDetector
        from mspoof_tts.sampling.hierarchical import HierarchicalDecoder

        logger.info("マルチ解像度スプーフ検出器をロードしています...")
        # 語彙サイズはトークナイザーから取得する（モデルロード後に上書きされる場合があるため、
        # ここではデフォルト値を使用し、モデルロード後に再初期化は不要な設計）
        detector_vocab_size = tts_model.tokenizer.vocab_size
        detectors = MultiResolutionDetector(
            vocab_size=detector_vocab_size,
        ).to(device)
        detectors.load_checkpoints(Path(args.checkpoint_dir))
        detectors.eval()

        hierarchical_decoder = HierarchicalDecoder(
            eas_sampler=eas_sampler,
            detectors=detectors,
            warmup=config.hierarchical.warmup_length,
            stage_lengths=config.hierarchical.stage_lengths,
            beam_sizes=config.hierarchical.beam_sizes,
            rank_weights=config.hierarchical.rank_weights,
            max_length=args.max_tokens,
        )
        logger.info("階層的デコーダーを初期化しました。")

    # ------------------------------------------------------------------
    # 3. 入力の準備
    # ------------------------------------------------------------------

    # 入力テキストの読み込み
    try:
        input_texts = load_input_text(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error("入力テキストの読み込みに失敗しました: %s", e)
        sys.exit(1)

    # リファレンステキストの読み込み
    try:
        ref_text = load_reference_text(args)
    except ValueError as e:
        logger.error("リファレンステキストの読み込みに失敗しました: %s", e)
        sys.exit(1)

    # リファレンス音声をエンコード
    logger.info("リファレンス音声をエンコードしています: %s", args.reference_audio)
    ref_codes = tts_model.encode_reference(args.reference_audio)
    logger.info("  リファレンストークン数: %d", len(ref_codes))

    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. テキストごとに音声を生成
    # ------------------------------------------------------------------
    for idx, text in enumerate(input_texts):
        logger.info("")
        logger.info("=" * 60)
        logger.info("文 %d / %d: %s", idx + 1, len(input_texts), text[:80])
        logger.info("=" * 60)

        if args.mode == "eas":
            # -------------------------------------------------------
            # EASモード: カスタム自己回帰ループで生成
            # -------------------------------------------------------

            # プロンプトの構築 (NeuTTS のチャットテンプレートを使用)
            logger.info("プロンプトを構築しています...")
            prompt_ids = tts_model._apply_chat_template(
                ref_codes=ref_codes,
                ref_text=ref_text,
                input_text=text,
            )
            logger.info("  プロンプトトークン数: %d", len(prompt_ids))

            # EASによるトークン生成
            logger.info("EAS によるトークン生成を開始します...")
            start_time = time.time()

            generated_ids = generate_with_eas(
                model=tts_model,
                prompt_ids=prompt_ids,
                eas_sampler=eas_sampler,
                max_new_tokens=args.max_tokens,
                device=device,
            )

            elapsed = time.time() - start_time
            logger.info(
                "生成完了: %d トークン (%.2f 秒, %.1f トークン/秒)",
                len(generated_ids),
                elapsed,
                len(generated_ids) / elapsed if elapsed > 0 else 0,
            )

            # トークンIDを speech 文字列に変換
            output_str = tokens_to_speech_string(tts_model, generated_ids)

        elif args.mode == "hierarchical":
            # -------------------------------------------------------
            # 階層的デコーディングモード: EAS + ビームサーチ + 枝刈り
            # -------------------------------------------------------

            # プロンプトの構築 (NeuTTS のチャットテンプレートを使用)
            logger.info("プロンプトを構築しています...")
            prompt_ids = tts_model._apply_chat_template(
                ref_codes=ref_codes,
                ref_text=ref_text,
                input_text=text,
            )
            logger.info("  プロンプトトークン数: %d", len(prompt_ids))

            # 階層的デコーディングによるトークン生成
            logger.info("階層的デコーディングによるトークン生成を開始します...")
            start_time = time.time()

            generated_ids = generate_with_hierarchical(
                model=tts_model,
                prompt_ids=prompt_ids,
                hierarchical_decoder=hierarchical_decoder,
                max_new_tokens=args.max_tokens,
                device=device,
            )

            elapsed = time.time() - start_time
            logger.info(
                "生成完了: %d トークン (%.2f 秒, %.1f トークン/秒)",
                len(generated_ids),
                elapsed,
                len(generated_ids) / elapsed if elapsed > 0 else 0,
            )

            # トークンIDを speech 文字列に変換
            output_str = tokens_to_speech_string(tts_model, generated_ids)

        # ----------------------------------------------------------
        # 5. トークン列を音声波形に変換して保存
        # ----------------------------------------------------------
        logger.info("トークン列を音声波形にデコードしています...")
        try:
            wav = tts_model._decode(output_str)
        except ValueError as e:
            logger.error(
                "デコードに失敗しました（有効なspeechトークンが見つかりません）: %s", e
            )
            continue

        # ウォーターマーク適用 (NeuTTS のウォーターマーカーが利用可能な場合)
        if tts_model.watermarker is not None:
            wav = tts_model.watermarker.apply_watermark(wav, sample_rate=24_000)

        # 出力ファイルパスの決定 (複数文の場合はインデックス付き)
        if len(input_texts) == 1:
            out_file = output_path
        else:
            stem = output_path.stem
            suffix = output_path.suffix or ".wav"
            out_file = output_path.parent / f"{stem}_{idx + 1:03d}{suffix}"

        # WAVファイルとして保存
        sf.write(str(out_file), wav, samplerate=tts_model.sample_rate)
        logger.info("音声ファイルを保存しました: %s", out_file)
        logger.info("  サンプルレート: %d Hz", tts_model.sample_rate)
        logger.info("  長さ: %.2f 秒", len(wav) / tts_model.sample_rate)

    logger.info("")
    logger.info("========================================")
    logger.info("全ての合成が完了しました。")
    logger.info("========================================")


if __name__ == "__main__":
    main()
