"""LibriTTSからの合成データ生成スクリプト。

スプーフ検出器の訓練に必要なreal/fakeペアデータセットを構築する。

手順:
    1. LibriTTS training splitの全発話に対し、NeuTTSで合成音声を生成
    2. 実音声と合成音声をNeuCodecで離散トークン列に変換
    3. real/fakeラベルを付与してメタデータファイルを生成

Usage:
    python -m mspoof_tts.data.prepare \\
        --libritts_dir /path/to/LibriTTS \\
        --output_dir /path/to/output \\
        --num_synthesis 3
"""

import json
import logging
import random
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def encode_audio_to_tokens(
    audio_path: Path,
    codec_model: object,
) -> list[int]:
    """音声ファイルをNeuCodecで離散トークン列に変換する。

    Args:
        audio_path: 入力音声ファイルのパス。
        codec_model: NeuCodecトークナイザーモデル。

    Returns:
        離散トークンIDのリスト。

    Raises:
        RuntimeError: 音声の読み込みまたはエンコードに失敗した場合。
    """
    try:
        # 16kHzモノラルで音声を読み込む
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        # torch tensorに変換: [1, 1, T] の形状にする
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

        # コーデックデバイスに転送してエンコードする
        if hasattr(codec_model, "device"):
            wav_tensor = wav_tensor.to(codec_model.device)

        with torch.no_grad():
            token_tensor = codec_model.encode_code(audio_or_path=wav_tensor)

        # 余分な次元を除去してリストに変換する
        tokens = token_tensor.squeeze().cpu().tolist()

        # スカラーの場合はリストに包む
        if isinstance(tokens, (int, float)):
            tokens = [int(tokens)]
        else:
            tokens = [int(t) for t in tokens]

        return tokens

    except Exception as e:
        raise RuntimeError(
            f"音声のトークン化に失敗しました: {audio_path}: {e}"
        ) from e


def synthesize_utterance(
    text: str,
    speaker_prompt_path: Path,
    tts_model: object,
    seed: Optional[int] = None,
) -> np.ndarray:
    """NeuTTSでテキストから音声を合成する。

    Args:
        text: 入力テキスト。
        speaker_prompt_path: 話者プロンプト音声のパス。
        tts_model: NeuTTSモデル。
        seed: ランダムシード（再現性のため）。

    Returns:
        合成された音声のnumpy配列（24kHz）。

    Raises:
        RuntimeError: 合成に失敗した場合。
    """
    # シードを設定する（再現性のため）
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        # リファレンス音声をエンコードする
        ref_codes = tts_model.encode_reference(str(speaker_prompt_path))

        # リファレンスのテキストを取得する（同じディレクトリにある.normalized.txtを探す）
        ref_text_path = speaker_prompt_path.with_suffix(".normalized.txt")
        if ref_text_path.exists():
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        else:
            # .normalized.txt がない場合は空文字列を使用する
            warnings.warn(
                f"リファレンステキストが見つかりません: {ref_text_path}。"
                "空文字列を使用します。"
            )
            ref_text = ""

        # NeuTTSで音声を合成する
        audio = tts_model.infer(text=text, ref_codes=ref_codes, ref_text=ref_text)

        return audio

    except Exception as e:
        raise RuntimeError(
            f"音声合成に失敗しました: text='{text[:50]}...', "
            f"ref={speaker_prompt_path}: {e}"
        ) from e


def _collect_speaker_utterances(
    libritts_dir: Path,
    splits: list[str],
) -> dict[str, list[dict]]:
    """LibriTTSディレクトリから話者ごとの発話情報を収集する。

    Args:
        libritts_dir: LibriTTSデータセットのルートディレクトリ。
        splits: 処理するsplit名のリスト。

    Returns:
        話者IDをキー、発話情報リストを値とする辞書。
        各発話情報は {"wav_path", "text_path", "utterance_id", "speaker_id",
        "chapter_id", "split"} を持つ。
    """
    speaker_utterances: dict[str, list[dict]] = {}

    for split_name in splits:
        split_dir = libritts_dir / split_name
        if not split_dir.exists():
            logger.warning("splitディレクトリが存在しません: %s", split_dir)
            continue

        # LibriTTS構造: split/speaker_id/chapter_id/speaker_chapter_utt.wav
        for wav_path in sorted(split_dir.glob("*/*/*.wav")):
            # .normalized.txt ファイルが存在するか確認する
            text_path = wav_path.with_suffix(".normalized.txt")
            if not text_path.exists():
                continue

            # ファイル名からID情報を抽出する
            stem = wav_path.stem  # e.g., "1234_5678_000001"
            parts = stem.split("_")
            if len(parts) < 3:
                continue

            speaker_id = parts[0]
            chapter_id = parts[1]
            utterance_id = stem

            if speaker_id not in speaker_utterances:
                speaker_utterances[speaker_id] = []

            speaker_utterances[speaker_id].append({
                "wav_path": wav_path,
                "text_path": text_path,
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "split": split_name,
            })

    return speaker_utterances


def _assign_splits(
    speaker_ids: list[str],
    seed: int = 42,
) -> dict[str, str]:
    """話者単位でtrain/val/testに分割する（90/5/5）。

    Args:
        speaker_ids: 全話者IDのリスト。
        seed: シャッフルのランダムシード。

    Returns:
        話者IDをキー、分割名("train", "val", "test")を値とする辞書。
    """
    rng = random.Random(seed)
    ids = sorted(speaker_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))

    split_map = {}
    for speaker_id in ids[:n_val]:
        split_map[speaker_id] = "val"
    for speaker_id in ids[n_val : n_val + n_test]:
        split_map[speaker_id] = "test"
    for speaker_id in ids[n_val + n_test :]:
        split_map[speaker_id] = "train"

    return split_map


def prepare_dataset(
    libritts_dir: Path,
    output_dir: Path,
    num_synthesis: int = 3,
    splits: Optional[list[str]] = None,
    backbone_repo: str = "neuphonic/neutts-nano",
    codec_repo: str = "neuphonic/neucodec",
    device: str = "cpu",
    seed_base: int = 42,
    resume: bool = False,
) -> None:
    """LibriTTSからreal/fakeペアデータセットを構築する。

    Args:
        libritts_dir: LibriTTSデータセットのルートディレクトリ。
        output_dir: 出力ディレクトリ。
        num_synthesis: 各発話あたりの合成数。
        splits: 処理する分割名のリスト。
            デフォルトは ["train-clean-100", "train-clean-360", "train-other-500"]。
        backbone_repo: NeuTTSバックボーンのリポジトリ名。
        codec_repo: コーデックのリポジトリ名。
        device: モデルを配置するデバイス。
        seed_base: 再現性のためのベースシード。
        resume: Trueの場合、既に処理済みのファイルをスキップする。
    """
    from neutts import NeuTTS

    if splits is None:
        splits = ["train-clean-100", "train-clean-360", "train-other-500"]

    libritts_dir = Path(libritts_dir)
    output_dir = Path(output_dir)

    # 出力ディレクトリ構造を作成する
    tokens_real_dir = output_dir / "tokens" / "real"
    tokens_fake_dir = output_dir / "tokens" / "fake"
    tokens_real_dir.mkdir(parents=True, exist_ok=True)
    tokens_fake_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    logger.info("NeuTTSモデルを初期化中...")
    tts_model = NeuTTS(
        backbone_repo=backbone_repo,
        codec_repo=codec_repo,
        backbone_device=device,
        codec_device=device,
    )
    codec_model = tts_model.codec

    # LibriTTSの発話情報を収集する
    logger.info("LibriTTSの発話情報を収集中...")
    speaker_utterances = _collect_speaker_utterances(libritts_dir, splits)

    if not speaker_utterances:
        logger.error(
            "発話が見つかりませんでした。LibriTTSのディレクトリパスを確認してください: %s",
            libritts_dir,
        )
        return

    total_speakers = len(speaker_utterances)
    total_utterances = sum(len(utts) for utts in speaker_utterances.values())
    logger.info(
        "発話情報の収集完了: %d 話者、%d 発話",
        total_speakers,
        total_utterances,
    )

    # 話者単位でtrain/val/testに分割する
    split_map = _assign_splits(list(speaker_utterances.keys()), seed=seed_base)

    # resumeモード: 既存のメタデータを読み込んで処理済みIDを取得する
    processed_ids: set[str] = set()
    if resume and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    processed_ids.add(entry["utterance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("レジュームモード: %d 件の処理済みエントリを検出", len(processed_ids))

    # メタデータファイルを開く（resumeモードの場合は追記、そうでない場合は上書き）
    metadata_mode = "a" if resume and metadata_path.exists() else "w"
    metadata_file = open(metadata_path, metadata_mode, encoding="utf-8")

    skipped_count = 0
    error_count = 0

    try:
        # 全話者の全発話をフラットなリストに展開して進捗バーを表示する
        all_utterances = []
        for speaker_id, utterances in speaker_utterances.items():
            for utt in utterances:
                all_utterances.append(utt)

        pbar = tqdm(all_utterances, desc="データセット構築中", unit="発話")

        for utt_info in pbar:
            utterance_id = utt_info["utterance_id"]
            speaker_id = utt_info["speaker_id"]
            wav_path = utt_info["wav_path"]
            text_path = utt_info["text_path"]
            data_split = split_map.get(speaker_id, "train")

            # resumeモード: realとfakeの全てが処理済みかチェックする
            if resume:
                real_id = f"{utterance_id}_real"
                fake_ids = [
                    f"{utterance_id}_fake_{i}" for i in range(num_synthesis)
                ]
                all_ids = [real_id] + fake_ids
                if all(uid in processed_ids for uid in all_ids):
                    skipped_count += 1
                    pbar.set_postfix(
                        skipped=skipped_count, errors=error_count
                    )
                    continue

            # テキストを読み込む
            try:
                text = text_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(
                    "テキスト読み込みに失敗 (%s): %s", text_path, e
                )
                error_count += 1
                pbar.set_postfix(skipped=skipped_count, errors=error_count)
                continue

            # --- 実音声のトークン化 ---
            real_token_filename = f"{utterance_id}_real.pt"
            real_token_path = tokens_real_dir / real_token_filename
            real_relative_path = f"tokens/real/{real_token_filename}"

            if resume and real_token_path.exists():
                # 既存のトークンファイルが存在する場合はスキップする
                pass
            else:
                try:
                    real_tokens = encode_audio_to_tokens(wav_path, codec_model)
                    token_tensor = torch.tensor(real_tokens, dtype=torch.long)
                    torch.save(token_tensor, real_token_path)
                except Exception as e:
                    logger.warning(
                        "実音声のトークン化に失敗 (%s): %s", wav_path, e
                    )
                    error_count += 1
                    pbar.set_postfix(
                        skipped=skipped_count, errors=error_count
                    )
                    continue

            # メタデータエントリを書き込む（real）
            real_entry_id = f"{utterance_id}_real"
            if real_entry_id not in processed_ids:
                real_entry = {
                    "utterance_id": real_entry_id,
                    "token_path": real_relative_path,
                    "label": 1,
                    "speaker_id": speaker_id,
                    "source_audio": str(wav_path),
                    "text": text,
                    "split": data_split,
                    "type": "real",
                }
                metadata_file.write(json.dumps(real_entry, ensure_ascii=False) + "\n")
                processed_ids.add(real_entry_id)

            # --- 合成音声の生成とトークン化 ---
            # 同一話者の別発話をリファレンスとして選択する
            speaker_utts = speaker_utterances[speaker_id]
            other_utts = [
                u for u in speaker_utts
                if u["utterance_id"] != utterance_id
            ]
            if not other_utts:
                # 同一話者の別発話がない場合は自身をリファレンスとする
                logger.warning(
                    "話者 %s には発話が1つしかありません。自身をリファレンスとして使用します。",
                    speaker_id,
                )
                other_utts = [utt_info]

            # リファレンスを決定的に選択する（ベースシードと発話IDに基づく）
            ref_rng = random.Random(seed_base + hash(utterance_id))
            ref_utt = ref_rng.choice(other_utts)
            ref_audio_path = ref_utt["wav_path"]

            synthesis_success = True
            for synth_idx in range(num_synthesis):
                fake_entry_id = f"{utterance_id}_fake_{synth_idx}"
                fake_token_filename = f"{utterance_id}_fake_{synth_idx}.pt"
                fake_token_path = tokens_fake_dir / fake_token_filename
                fake_relative_path = f"tokens/fake/{fake_token_filename}"

                # resumeモード: 既に処理済みならスキップする
                if resume and fake_entry_id in processed_ids:
                    continue

                # 各合成ごとに異なるシードを使用する
                synth_seed = seed_base + hash(utterance_id) + synth_idx

                try:
                    # NeuTTSで合成する
                    synth_audio = synthesize_utterance(
                        text=text,
                        speaker_prompt_path=ref_audio_path,
                        tts_model=tts_model,
                        seed=synth_seed,
                    )

                    # 合成音声を一時ファイルに保存してからトークン化する
                    # NeuTTSの出力は24kHz、NeuCodecのエンコードは16kHzを期待するため
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        sf.write(
                            str(tmp_path),
                            synth_audio,
                            samplerate=tts_model.sample_rate,
                        )

                    try:
                        fake_tokens = encode_audio_to_tokens(
                            tmp_path, codec_model
                        )
                        token_tensor = torch.tensor(
                            fake_tokens, dtype=torch.long
                        )
                        torch.save(token_tensor, fake_token_path)
                    finally:
                        # 一時ファイルを削除する
                        tmp_path.unlink(missing_ok=True)

                    # メタデータエントリを書き込む（fake）
                    fake_entry = {
                        "utterance_id": fake_entry_id,
                        "token_path": fake_relative_path,
                        "label": 0,
                        "speaker_id": speaker_id,
                        "source_audio": str(wav_path),
                        "reference_audio": str(ref_audio_path),
                        "text": text,
                        "split": data_split,
                        "type": "fake",
                        "synthesis_index": synth_idx,
                        "seed": synth_seed,
                    }
                    metadata_file.write(
                        json.dumps(fake_entry, ensure_ascii=False) + "\n"
                    )
                    processed_ids.add(fake_entry_id)

                except Exception as e:
                    logger.warning(
                        "合成 #%d に失敗 (%s): %s",
                        synth_idx,
                        utterance_id,
                        e,
                    )
                    error_count += 1
                    synthesis_success = False

            # メタデータを定期的にフラッシュする
            metadata_file.flush()

            pbar.set_postfix(skipped=skipped_count, errors=error_count)

    finally:
        metadata_file.close()

    # 最終統計をログに出力する
    logger.info("データセット構築完了:")
    logger.info("  出力ディレクトリ: %s", output_dir)
    logger.info("  メタデータファイル: %s", metadata_path)
    logger.info("  処理済みエントリ数: %d", len(processed_ids))
    logger.info("  スキップ数: %d", skipped_count)
    logger.info("  エラー数: %d", error_count)

    # 分割ごとの統計を出力する
    split_counts: dict[str, dict[str, int]] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                s = entry.get("split", "unknown")
                t = entry.get("type", "unknown")
                if s not in split_counts:
                    split_counts[s] = {"real": 0, "fake": 0}
                split_counts[s][t] = split_counts[s].get(t, 0) + 1
            except (json.JSONDecodeError, KeyError):
                continue

    for split_name, counts in sorted(split_counts.items()):
        logger.info(
            "  %s: real=%d, fake=%d",
            split_name,
            counts.get("real", 0),
            counts.get("fake", 0),
        )


if __name__ == "__main__":
    import argparse

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="LibriTTSからスプーフ検出器訓練用データセットを構築する"
    )
    parser.add_argument(
        "--libritts_dir", type=Path, required=True, help="LibriTTSのルートディレクトリ"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="出力ディレクトリ"
    )
    parser.add_argument(
        "--num_synthesis", type=int, default=3, help="各発話あたりの合成数"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-nano",
        help="NeuTTSバックボーンのリポジトリ名 (default: neuphonic/neutts-nano)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="neuphonic/neucodec",
        help="コーデックのリポジトリ名 (default: neuphonic/neucodec)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="モデルを配置するデバイス (default: cpu)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="処理するsplit名のカンマ区切りリスト (例: train-clean-100,train-clean-360)",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="再現性のためのベースシード (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="処理済みファイルをスキップしてレジュームする",
    )
    args = parser.parse_args()

    # splitsのパースを行う
    parsed_splits = None
    if args.splits is not None:
        parsed_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    prepare_dataset(
        libritts_dir=args.libritts_dir,
        output_dir=args.output_dir,
        num_synthesis=args.num_synthesis,
        splits=parsed_splits,
        backbone_repo=args.backbone,
        codec_repo=args.codec,
        device=args.device,
        seed_base=args.seed_base,
        resume=args.resume,
    )
