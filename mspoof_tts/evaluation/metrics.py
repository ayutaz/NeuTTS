"""評価指標の計算モジュール。

以下の評価指標を計算する:
    - WER (Word Error Rate): Whisper-large-v3による音声認識精度
    - SIM (Speaker Similarity): WavLM-base-plus-svによる話者類似度
    - NISQA: 知覚品質スコア
    - MOSNET: 平均オピニオンスコア (MOS) の推定値
"""

from pathlib import Path

import torch


def compute_wer(
    generated_audio_path: Path,
    reference_text: str,
    whisper_model: object,
) -> float:
    """Whisper-large-v3を使用してWERを計算する。

    生成音声をWhisperで書き起こし、参照テキストとのWERを算出する。

    Args:
        generated_audio_path: 生成音声ファイルのパス。
        reference_text: 参照テキスト。
        whisper_model: Whisperモデル。

    Returns:
        WER値（0.0 ~ 1.0+）。低いほど良い。
    """
    raise NotImplementedError


def compute_similarity(
    generated_audio_path: Path,
    reference_audio_path: Path,
    wavlm_model: object,
) -> float:
    """WavLM-base-plus-svを使用して話者類似度を計算する。

    生成音声と参照音声の話者埋め込みのコサイン類似度を算出する。

    Args:
        generated_audio_path: 生成音声ファイルのパス。
        reference_audio_path: 参照話者音声ファイルのパス。
        wavlm_model: WavLMモデル。

    Returns:
        コサイン類似度 (-1.0 ~ 1.0)。高いほど良い。
    """
    raise NotImplementedError


def compute_nisqa(
    audio_path: Path,
    nisqa_model: object,
) -> float:
    """NISQAモデルで知覚品質スコアを計算する。

    Args:
        audio_path: 評価対象の音声ファイルのパス。
        nisqa_model: NISQAモデル。

    Returns:
        NISQAスコア。高いほど良い。
    """
    raise NotImplementedError


def compute_mosnet(
    audio_path: Path,
    mosnet_model: object,
) -> float:
    """MOSNetモデルで平均オピニオンスコアを推定する。

    Args:
        audio_path: 評価対象の音声ファイルのパス。
        mosnet_model: MOSNetモデル。

    Returns:
        推定MOSスコア (1.0 ~ 5.0)。高いほど良い。
    """
    raise NotImplementedError
