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

from pathlib import Path
from typing import Optional


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
    """
    raise NotImplementedError


def synthesize_utterance(
    text: str,
    speaker_prompt_path: Path,
    tts_model: object,
    seed: Optional[int] = None,
) -> Path:
    """NeuTTSでテキストから音声を合成する。

    Args:
        text: 入力テキスト。
        speaker_prompt_path: 話者プロンプト音声のパス。
        tts_model: NeuTTSモデル。
        seed: ランダムシード（再現性のため）。

    Returns:
        合成された音声ファイルのパス。
    """
    raise NotImplementedError


def prepare_dataset(
    libritts_dir: Path,
    output_dir: Path,
    num_synthesis: int = 3,
    splits: Optional[list[str]] = None,
) -> None:
    """LibriTTSからreal/fakeペアデータセットを構築する。

    Args:
        libritts_dir: LibriTTSデータセットのルートディレクトリ。
        output_dir: 出力ディレクトリ。
        num_synthesis: 各発話あたりの合成数。
        splits: 処理する分割名のリスト。
            デフォルトは ["train-clean-100", "train-clean-360", "train-other-500"]。
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    prepare_dataset(
        libritts_dir=args.libritts_dir,
        output_dir=args.output_dir,
        num_synthesis=args.num_synthesis,
    )
