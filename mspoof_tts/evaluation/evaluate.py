"""評価パイプライン。

ベースライン（NeuTTS標準デコーディング）と提案手法（MSpoof-TTS）で
合成された音声を各評価指標で評価し、結果を集計する。

評価データセット:
    - LibriSpeech
    - LibriTTS
    - TwistList

Usage:
    python -m mspoof_tts.evaluation.evaluate \\
        --generated_dir /path/to/generated \\
        --reference_dir /path/to/reference \\
        --output_path results.json
"""

from pathlib import Path
from typing import Optional

import torch

from mspoof_tts.evaluation.metrics import (
    compute_wer,
    compute_similarity,
    compute_nisqa,
    compute_mosnet,
)


class EvaluationPipeline:
    """評価パイプライン。

    生成音声と参照データを受け取り、全評価指標を計算して結果を集計する。

    Args:
        whisper_model_name: Whisperモデル名。
        wavlm_model_name: WavLMモデル名。
        device: 計算デバイス。
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-large-v3",
        wavlm_model_name: str = "microsoft/wavlm-base-plus-sv",
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError("EvaluationPipeline の実装が必要")

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
        raise NotImplementedError

    def evaluate_dataset(
        self,
        generated_dir: Path,
        reference_dir: Path,
        output_path: Optional[Path] = None,
    ) -> dict[str, float]:
        """データセット全体の評価を実行し、平均スコアを返す。

        Args:
            generated_dir: 生成音声のディレクトリ。
            reference_dir: 参照データのディレクトリ。
            output_path: 結果のJSON出力パス（Noneの場合は保存しない）。

        Returns:
            各指標の平均値を含む辞書。
        """
        raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSpoof-TTS評価パイプライン")
    parser.add_argument(
        "--generated_dir", type=Path, required=True, help="生成音声のディレクトリ"
    )
    parser.add_argument(
        "--reference_dir", type=Path, required=True, help="参照データのディレクトリ"
    )
    parser.add_argument(
        "--output_path", type=Path, default=None, help="結果の出力パス (JSON)"
    )
    args = parser.parse_args()

    pipeline = EvaluationPipeline()
    results = pipeline.evaluate_dataset(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        output_path=args.output_path,
    )
    print(results)
