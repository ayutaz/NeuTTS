"""評価パイプライン。

ベースライン（NeuTTS標準デコーディング）と提案手法（MSpoof-TTS）で
合成された音声を各評価指標で評価し、結果を集計する。

評価データセット:
    - LibriSpeech
    - LibriTTS
    - TwistList

Usage:
    python -m mspoof_tts.evaluation.evaluate \
        --generated_dir /path/to/generated \
        --reference_dir /path/to/reference \
        --output_path results.json

    # evaluate.sh からの呼び出し:
    python -m mspoof_tts.evaluation.evaluate \
        --metric wer \
        --synth-dir /path/to/synth \
        --results-dir /path/to/results \
        --dataset libritts
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from mspoof_tts.evaluation.metrics import (
    compute_wer,
    compute_similarity,
    compute_nisqa,
    compute_mosnet,
    load_whisper_model,
    load_wavlm_model,
    load_nisqa_model,
    load_mosnet_model,
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """評価パイプライン。

    生成音声と参照データを受け取り、全評価指標を計算して結果を集計する。

    Args:
        whisper_model_name: Whisperモデル名。
        wavlm_model_name: WavLMモデル名。
        device: 計算デバイス。
        lazy_load: Trueの場合、モデルを初回使用時に遅延ロードする。
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-large-v3",
        wavlm_model_name: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cuda",
        lazy_load: bool = True,
    ) -> None:
        """モデルの初期化を行う。

        Args:
            whisper_model_name: Whisperモデル名またはパス。
            wavlm_model_name: WavLMモデル名またはパス。
            device: 計算デバイス（"cuda" または "cpu"）。
            lazy_load: Trueの場合、モデルを初回使用時にロードする。
        """
        self.whisper_model_name = whisper_model_name
        self.wavlm_model_name = wavlm_model_name
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

        # 遅延ロード用のプレースホルダー
        self._whisper_model = None
        self._wavlm_model = None
        self._nisqa_model = None
        self._mosnet_model = None

        if not lazy_load:
            logger.info("全モデルを即座にロードします...")
            self._load_all_models()

    def _load_all_models(self) -> None:
        """全モデルをロードする。"""
        _ = self.whisper_model
        _ = self.wavlm_model
        _ = self.nisqa_model
        _ = self.mosnet_model

    @property
    def whisper_model(self) -> tuple:
        """Whisperモデルとプロセッサのタプルを取得する（遅延ロード対応）。"""
        if self._whisper_model is None:
            logger.info("Whisperモデルをロード中: %s", self.whisper_model_name)
            self._whisper_model = load_whisper_model(
                model_name=self.whisper_model_name, device=self.device
            )
        return self._whisper_model

    @property
    def wavlm_model(self) -> tuple:
        """WavLMモデルと特徴量抽出器のタプルを取得する（遅延ロード対応）。"""
        if self._wavlm_model is None:
            logger.info("WavLMモデルをロード中: %s", self.wavlm_model_name)
            self._wavlm_model = load_wavlm_model(
                model_name=self.wavlm_model_name, device=self.device
            )
        return self._wavlm_model

    @property
    def nisqa_model(self) -> object:
        """NISQAモデルを取得する（遅延ロード対応）。"""
        if self._nisqa_model is None:
            logger.info("NISQAモデルをロード中...")
            self._nisqa_model = load_nisqa_model(device=self.device)
        return self._nisqa_model

    @property
    def mosnet_model(self) -> object:
        """MOSNetモデルを取得する（遅延ロード対応）。"""
        if self._mosnet_model is None:
            logger.info("MOSNetモデルをロード中...")
            self._mosnet_model = load_mosnet_model(device=self.device)
        return self._mosnet_model

    def evaluate_single(
        self,
        generated_audio_path: Path,
        reference_text: str,
        reference_audio_path: Path,
    ) -> dict[str, float]:
        """単一発話の評価を実行する。

        Args:
            generated_audio_path: 生成音声ファイルのパス。
            reference_text: 参照テキスト。
            reference_audio_path: 参照話者音声ファイルのパス。

        Returns:
            評価指標の辞書 {"wer": ..., "sim": ..., "nisqa": ..., "mosnet": ...}。
        """
        results: dict[str, float] = {}

        # WER
        whisper_model, whisper_processor = self.whisper_model
        wer_value = compute_wer(
            generated_audio_path=generated_audio_path,
            reference_text=reference_text,
            whisper_model=whisper_model,
            whisper_processor=whisper_processor,
        )
        results["wer"] = wer_value

        # 話者類似度 (SIM)
        wavlm_model, wavlm_feature_extractor = self.wavlm_model
        sim_value = compute_similarity(
            generated_audio_path=generated_audio_path,
            reference_audio_path=reference_audio_path,
            wavlm_model=wavlm_model,
            wavlm_feature_extractor=wavlm_feature_extractor,
        )
        results["sim"] = sim_value

        # NISQA
        nisqa_value = compute_nisqa(
            audio_path=generated_audio_path,
            nisqa_model=self.nisqa_model,
        )
        results["nisqa"] = nisqa_value

        # MOSNET
        mosnet_value = compute_mosnet(
            audio_path=generated_audio_path,
            mosnet_model=self.mosnet_model,
        )
        results["mosnet"] = mosnet_value

        return results

    def evaluate_single_metric(
        self,
        metric: str,
        generated_audio_path: Path,
        reference_text: Optional[str] = None,
        reference_audio_path: Optional[Path] = None,
    ) -> dict[str, float]:
        """単一発話に対して指定された1つの指標のみを計算する。

        Args:
            metric: 評価指標名（"wer", "sim", "nisqa", "mosnet"）。
            generated_audio_path: 生成音声ファイルのパス。
            reference_text: 参照テキスト（WER計算時に必要）。
            reference_audio_path: 参照話者音声ファイルのパス（SIM計算時に必要）。

        Returns:
            指定された指標の値を含む辞書。
        """
        results: dict[str, float] = {}

        if metric == "wer":
            if reference_text is None:
                logger.warning(
                    "参照テキストが未指定のため、WERをスキップ: %s",
                    generated_audio_path.name,
                )
                return results
            whisper_model, whisper_processor = self.whisper_model
            results["wer"] = compute_wer(
                generated_audio_path=generated_audio_path,
                reference_text=reference_text,
                whisper_model=whisper_model,
                whisper_processor=whisper_processor,
            )
        elif metric == "sim":
            if reference_audio_path is None:
                logger.warning(
                    "参照音声が未指定のため、SIMをスキップ: %s",
                    generated_audio_path.name,
                )
                return results
            wavlm_model, wavlm_feature_extractor = self.wavlm_model
            results["sim"] = compute_similarity(
                generated_audio_path=generated_audio_path,
                reference_audio_path=reference_audio_path,
                wavlm_model=wavlm_model,
                wavlm_feature_extractor=wavlm_feature_extractor,
            )
        elif metric == "nisqa":
            results["nisqa"] = compute_nisqa(
                audio_path=generated_audio_path,
                nisqa_model=self.nisqa_model,
            )
        elif metric == "mosnet":
            results["mosnet"] = compute_mosnet(
                audio_path=generated_audio_path,
                mosnet_model=self.mosnet_model,
            )
        else:
            logger.warning("不明な評価指標: %s", metric)

        return results

    def evaluate_dataset(
        self,
        generated_dir: Path,
        reference_dir: Path,
        output_path: Optional[Path] = None,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """データセット全体の評価を実行し、平均スコアを返す。

        generated_dirに含まれる全wavファイルを走査し、reference_dir内の
        同名ファイル（.wav / .txt）とペアリングして評価指標を計算する。

        Args:
            generated_dir: 生成音声のディレクトリ。
            reference_dir: 参照データのディレクトリ。
            output_path: 結果のJSON出力パス（Noneの場合は保存しない）。
            metrics: 計算する指標のリスト。Noneの場合は全指標を計算する。

        Returns:
            各指標の平均値を含む辞書。
        """
        generated_dir = Path(generated_dir)
        reference_dir = Path(reference_dir)

        # 生成音声ファイルの一覧を取得
        generated_files = sorted(generated_dir.glob("*.wav"))
        if not generated_files:
            logger.warning("生成音声ファイルが見つかりません: %s", generated_dir)
            return {}

        logger.info(
            "評価開始: %d 件の生成音声ファイル（%s）",
            len(generated_files),
            generated_dir,
        )

        all_metrics = metrics or ["wer", "sim", "nisqa", "mosnet"]
        per_utterance_results: list[dict] = []
        metric_accumulators: dict[str, list[float]] = {m: [] for m in all_metrics}

        for gen_path in tqdm(generated_files, desc="評価中"):
            stem = gen_path.stem

            # 参照ファイルのパスを構築
            ref_audio_path = reference_dir / f"{stem}.wav"
            ref_text_path = reference_dir / f"{stem}.txt"

            # 参照テキストの読み込み
            reference_text: Optional[str] = None
            if ref_text_path.exists():
                reference_text = ref_text_path.read_text(encoding="utf-8").strip()
            elif "wer" in all_metrics:
                logger.warning(
                    "参照テキストが見つかりません（WERスキップ）: %s", ref_text_path
                )

            # 参照音声の存在確認
            ref_audio_exists = ref_audio_path.exists()
            if not ref_audio_exists and "sim" in all_metrics:
                logger.warning(
                    "参照音声が見つかりません（SIMスキップ）: %s", ref_audio_path
                )

            utterance_result: dict[str, object] = {"utterance_id": stem}

            if metrics is not None:
                # 特定の指標のみ計算
                for metric in all_metrics:
                    single_result = self.evaluate_single_metric(
                        metric=metric,
                        generated_audio_path=gen_path,
                        reference_text=reference_text,
                        reference_audio_path=ref_audio_path if ref_audio_exists else None,
                    )
                    for key, value in single_result.items():
                        utterance_result[key] = value
                        metric_accumulators[key].append(value)
            else:
                # 全指標を計算（参照データの有無に応じて個別に処理）
                needs_ref_text = reference_text is not None
                needs_ref_audio = ref_audio_exists

                if needs_ref_text and needs_ref_audio:
                    # 全指標を一括計算
                    single_results = self.evaluate_single(
                        generated_audio_path=gen_path,
                        reference_text=reference_text,
                        reference_audio_path=ref_audio_path,
                    )
                    for key, value in single_results.items():
                        utterance_result[key] = value
                        metric_accumulators[key].append(value)
                else:
                    # 参照データが不完全な場合、利用可能な指標のみ計算
                    for metric in all_metrics:
                        single_result = self.evaluate_single_metric(
                            metric=metric,
                            generated_audio_path=gen_path,
                            reference_text=reference_text,
                            reference_audio_path=ref_audio_path if ref_audio_exists else None,
                        )
                        for key, value in single_result.items():
                            utterance_result[key] = value
                            metric_accumulators[key].append(value)

            per_utterance_results.append(utterance_result)

        # 平均スコアの算出
        aggregate_results: dict[str, float] = {}
        for metric_name, values in metric_accumulators.items():
            if values:
                aggregate_results[f"avg_{metric_name}"] = sum(values) / len(values)
                aggregate_results[f"count_{metric_name}"] = len(values)
            else:
                logger.warning("指標 '%s' の有効な結果がありません。", metric_name)

        # 結果の保存
        full_results = {
            "aggregate": aggregate_results,
            "per_utterance": per_utterance_results,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 既存の結果ファイルがあればマージする
            if output_path.exists():
                try:
                    existing = json.loads(output_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and "aggregate" in existing:
                        existing["aggregate"].update(aggregate_results)
                        # 発話ごとの結果もマージ
                        existing_ids = {
                            r["utterance_id"]
                            for r in existing.get("per_utterance", [])
                        }
                        for r in per_utterance_results:
                            if r["utterance_id"] in existing_ids:
                                # 既存エントリを更新
                                for er in existing["per_utterance"]:
                                    if er["utterance_id"] == r["utterance_id"]:
                                        er.update(r)
                                        break
                            else:
                                existing["per_utterance"].append(r)
                        full_results = existing
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        "既存の結果ファイルの読み込みに失敗。上書きします: %s",
                        output_path,
                    )

            output_path.write_text(
                json.dumps(full_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("結果を保存しました: %s", output_path)

        # サマリーテーブルの表示
        _print_summary_table(aggregate_results)

        return aggregate_results


def _print_summary_table(aggregate_results: dict[str, float]) -> None:
    """評価結果のサマリーテーブルを標準出力に表示する。

    Args:
        aggregate_results: 集計済みの評価結果辞書。
    """
    metric_display = {
        "avg_wer": ("WER", "低いほど良い"),
        "avg_sim": ("SIM", "高いほど良い"),
        "avg_nisqa": ("NISQA", "高いほど良い"),
        "avg_mosnet": ("MOSNET", "高いほど良い"),
    }

    print("\n" + "=" * 50)
    print("  評価結果サマリー")
    print("=" * 50)
    print(f"  {'指標':<10} {'スコア':>10}   {'件数':>6}   {'備考'}")
    print("-" * 50)

    for key, (name, note) in metric_display.items():
        if key in aggregate_results:
            count_key = key.replace("avg_", "count_")
            count = int(aggregate_results.get(count_key, 0))
            print(f"  {name:<10} {aggregate_results[key]:>10.4f}   {count:>6}   {note}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MSpoof-TTS評価パイプライン")

    # 元のインターフェース引数
    parser.add_argument(
        "--generated_dir",
        type=Path,
        default=None,
        help="生成音声のディレクトリ",
    )
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=None,
        help="参照データのディレクトリ",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="結果の出力パス (JSON)",
    )

    # evaluate.sh からのインターフェース引数
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["wer", "sim", "nisqa", "mosnet"],
        help="計算する評価指標（単一指標モード）",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=None,
        help="合成音声ディレクトリ（--generated_dir の別名）",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        dest="reference_dir_hyphen",
        help="参照データのディレクトリ（--reference_dir の別名）",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="評価結果の保存先ディレクトリ",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="評価データセット名（結果ファイル名に使用）",
    )

    # 共通オプション
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="計算デバイス（デフォルト: cuda）",
    )

    args = parser.parse_args()

    # 引数の統合: --synth-dir は --generated_dir の別名として扱う
    generated_dir: Optional[Path] = args.generated_dir or args.synth_dir
    reference_dir: Optional[Path] = args.reference_dir or args.reference_dir_hyphen

    if generated_dir is None:
        parser.error("--generated_dir または --synth-dir を指定してください。")

    # reference_dirが未指定の場合、generated_dirの親ディレクトリに
    # referenceディレクトリがあると仮定する
    if reference_dir is None:
        # デフォルトとして generated_dir と同階層の reference/ を使用
        candidate = generated_dir.parent / "reference"
        if candidate.exists():
            reference_dir = candidate
            logger.info("参照ディレクトリを自動検出: %s", reference_dir)
        else:
            # 参照ディレクトリなしで動作（WER/SIMはスキップされる）
            reference_dir = generated_dir
            logger.warning(
                "参照ディレクトリが未指定です。WER/SIMの計算には参照データが必要です。"
            )

    # 出力パスの決定
    output_path: Optional[Path] = args.output_path
    if output_path is None and args.results_dir is not None:
        dataset_name = args.dataset or "eval"
        output_path = args.results_dir / f"{dataset_name}_results.json"

    # 計算する指標の決定
    metrics: Optional[list[str]] = None
    if args.metric is not None:
        metrics = [args.metric]

    # パイプラインの実行
    pipeline = EvaluationPipeline(device=args.device)
    results = pipeline.evaluate_dataset(
        generated_dir=generated_dir,
        reference_dir=reference_dir,
        output_path=output_path,
        metrics=metrics,
    )
