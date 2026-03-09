"""評価指標の計算モジュール。

以下の評価指標を計算する:
    - WER (Word Error Rate): Whisper-large-v3による音声認識精度
    - SIM (Speaker Similarity): WavLM-base-plus-svによる話者類似度
    - NISQA: 知覚品質スコア
    - MOSNET: 平均オピニオンスコア (MOS) の推定値
"""

import logging
import math
import re
import string
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# 音声ロードユーティリティ
# ---------------------------------------------------------------------------


def _load_audio(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """音声ファイルを読み込み、モノラル・指定サンプルレートに変換する。

    Args:
        audio_path: 音声ファイルのパス。
        target_sr: 目標サンプルレート。

    Returns:
        shape (1, T) のテンソル。

    Raises:
        FileNotFoundError: 音声ファイルが存在しない場合。
        RuntimeError: 音声が空（サンプル数0）の場合。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    # soundfile で読み込み（torchaudio の torchcodec 依存を回避）
    data, sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data)

    # モノラル化: (samples,) or (samples, channels)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    else:
        # (samples, channels) → (channels, samples) → mean → (1, T)
        waveform = waveform.T.mean(dim=0, keepdim=True)

    # リサンプリング
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    if waveform.shape[1] == 0:
        raise RuntimeError(f"音声ファイルが空です: {audio_path}")

    return waveform


def _normalize_text(text: str) -> str:
    """テキストを正規化する（小文字化、句読点除去、余分な空白の除去）。

    Args:
        text: 正規化対象のテキスト。

    Returns:
        正規化されたテキスト。
    """
    text = text.lower().strip()
    # 句読点を除去
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 連続する空白を1つに
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# モデルロードヘルパー
# ---------------------------------------------------------------------------


def load_whisper_model(
    model_name: str = "openai/whisper-large-v3",
    device: str = "cuda",
) -> tuple:
    """Whisperモデルとプロセッサをロードする。

    Args:
        model_name: HuggingFaceモデル名またはパス。
        device: 計算デバイス（"cuda" または "cpu"）。

    Returns:
        (model, processor) のタプル。
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def load_wavlm_model(
    model_name: str = "microsoft/wavlm-base-plus-sv",
    device: str = "cuda",
) -> tuple:
    """WavLMモデルと特徴量抽出器をロードする。

    Args:
        model_name: HuggingFaceモデル名またはパス。
        device: 計算デバイス（"cuda" または "cpu"）。

    Returns:
        (model, feature_extractor) のタプル。
    """
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, feature_extractor


def load_nisqa_model(device: str = "cuda") -> object | None:
    """NISQAモデルをロードする。利用不可の場合はNoneを返す。

    Args:
        device: 計算デバイス。

    Returns:
        NISQAモデルオブジェクト。ライブラリが利用不可の場合はNone。
    """
    try:
        import nisqa

        model = nisqa.NISQA(device=device)
        return model
    except ImportError:
        logger.warning(
            "nisqaライブラリがインストールされていません。"
            "NISQAスコアはフォールバック推定値を使用します。"
        )
        return None
    except Exception as e:
        logger.warning(f"NISQAモデルのロードに失敗しました: {e}")
        return None


def load_mosnet_model(device: str = "cuda") -> object | None:
    """MOSNetモデルをロードする。利用不可の場合はNoneを返す。

    Args:
        device: 計算デバイス。

    Returns:
        MOSNetモデルオブジェクト。ライブラリが利用不可の場合はNone。
    """
    try:
        import mosnet

        model = mosnet.MOSNet(device=device)
        return model
    except ImportError:
        logger.warning(
            "mosnetライブラリがインストールされていません。"
            "MOSNetスコアはフォールバック推定値を使用します。"
        )
        return None
    except Exception as e:
        logger.warning(f"MOSNetモデルのロードに失敗しました: {e}")
        return None


# ---------------------------------------------------------------------------
# 評価指標の計算
# ---------------------------------------------------------------------------


def compute_wer(
    generated_audio_path: Path,
    reference_text: str,
    whisper_model: object,
    whisper_processor: object,
) -> float:
    """Whisper-large-v3を使用してWERを計算する。

    生成音声をWhisperで書き起こし、参照テキストとのWERを算出する。

    Args:
        generated_audio_path: 生成音声ファイルのパス。
        reference_text: 参照テキスト。
        whisper_model: ロード済みWhisperモデル。
        whisper_processor: ロード済みWhisperプロセッサ。

    Returns:
        WER値（0.0 ~ 1.0+）。低いほど良い。
        参照テキストが空の場合はNaNを返す。
    """
    import jiwer

    reference_text = reference_text.strip()
    if not reference_text:
        logger.warning("参照テキストが空です。WERはNaNを返します。")
        return float("nan")

    # 音声のロード
    waveform = _load_audio(generated_audio_path, target_sr=TARGET_SAMPLE_RATE)

    # デバイスの取得
    device = next(whisper_model.parameters()).device

    # プロセッサで入力特徴量を作成
    input_features = whisper_processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(device)

    # 推論
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)

    # デコード
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    # テキストの正規化
    normalized_ref = _normalize_text(reference_text)
    normalized_hyp = _normalize_text(transcription)

    # 書き起こしが空の場合
    if not normalized_hyp:
        logger.warning("Whisperの書き起こし結果が空です。WERは1.0を返します。")
        return 1.0

    # WER計算
    wer_score = jiwer.wer(normalized_ref, normalized_hyp)
    return float(wer_score)


def compute_similarity(
    generated_audio_path: Path,
    reference_audio_path: Path,
    wavlm_model: object,
    wavlm_feature_extractor: object,
) -> float:
    """WavLM-base-plus-svを使用して話者類似度を計算する。

    生成音声と参照音声の話者埋め込みのコサイン類似度を算出する。

    Args:
        generated_audio_path: 生成音声ファイルのパス。
        reference_audio_path: 参照話者音声ファイルのパス。
        wavlm_model: ロード済みWavLMモデル。
        wavlm_feature_extractor: ロード済みWav2Vec2FeatureExtractor。

    Returns:
        コサイン類似度 (-1.0 ~ 1.0)。高いほど良い。
    """
    # 音声のロード
    gen_waveform = _load_audio(generated_audio_path, target_sr=TARGET_SAMPLE_RATE)
    ref_waveform = _load_audio(reference_audio_path, target_sr=TARGET_SAMPLE_RATE)

    device = next(wavlm_model.parameters()).device

    def _extract_embedding(waveform: torch.Tensor) -> torch.Tensor:
        """単一音声波形から話者埋め込みを抽出する。"""
        inputs = wavlm_feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = wavlm_model(input_values)
            embedding = outputs.embeddings  # shape: (1, embedding_dim)

        # L2正規化
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    gen_embedding = _extract_embedding(gen_waveform)
    ref_embedding = _extract_embedding(ref_waveform)

    # コサイン類似度（L2正規化済みなので内積で十分）
    similarity = torch.sum(gen_embedding * ref_embedding, dim=-1).item()
    return float(similarity)


# ---------------------------------------------------------------------------
# フォールバック推定
# ---------------------------------------------------------------------------


def _nisqa_spectral_fallback(audio_path: Path) -> float:
    """NISQAライブラリが利用不可の場合のスペクトル特徴量ベースの品質推定。

    単純なスペクトル特徴量（スペクトル平坦度、SNR推定など）を用いて
    おおまかな品質スコアを推定する。あくまでフォールバック用。

    Args:
        audio_path: 評価対象の音声ファイルのパス。

    Returns:
        推定品質スコア (1.0 ~ 5.0)。精度は限定的。
    """
    logger.warning(
        "NISQAフォールバック: スペクトル特徴量ベースの簡易推定を使用します。"
        "結果は参考値です。"
    )

    waveform = _load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)
    audio = waveform.squeeze(0)  # shape: (T,)

    # スペクトログラム計算
    n_fft = 512
    hop_length = 160
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    mag = spec.abs()  # shape: (freq_bins, time_frames)

    # スペクトル平坦度 (Spectral Flatness)
    # 幾何平均 / 算術平均。1に近いほど白色雑音（低品質の可能性）
    log_mag = torch.log(mag + 1e-10)
    geometric_mean = torch.exp(log_mag.mean(dim=0))
    arithmetic_mean = mag.mean(dim=0)
    spectral_flatness = (geometric_mean / (arithmetic_mean + 1e-10)).mean().item()

    # 簡易SNR推定: 信号パワーと低エネルギーフレームのパワー比
    frame_energy = mag.pow(2).sum(dim=0)
    sorted_energy, _ = frame_energy.sort()
    n_frames = sorted_energy.shape[0]
    noise_floor = sorted_energy[: max(1, n_frames // 10)].mean()
    signal_power = sorted_energy[n_frames // 2 :].mean()
    snr_estimate = 10 * math.log10(
        (signal_power / (noise_floor + 1e-10)).clamp(min=1e-10).item()
    )

    # 有声区間の割合（零交差率ベースの簡易推定）
    zero_crossings = ((audio[:-1] * audio[1:]) < 0).float().mean().item()

    # ヒューリスティックなスコアマッピング (1.0 ~ 5.0)
    # - 高SNR → 高スコア
    # - 低スペクトル平坦度 → 音声らしい → 高スコア
    # - 適度な零交差率 → 高スコア
    snr_score = min(max((snr_estimate - 5) / 30.0, 0.0), 1.0)  # 5~35dB → 0~1
    flatness_score = 1.0 - min(spectral_flatness, 1.0)  # 低いほど良い
    zcr_score = 1.0 - abs(zero_crossings - 0.1) / 0.2  # 0.1付近が最適
    zcr_score = min(max(zcr_score, 0.0), 1.0)

    # 重み付き結合
    raw_score = 0.5 * snr_score + 0.3 * flatness_score + 0.2 * zcr_score
    quality_score = 1.0 + raw_score * 4.0  # 1.0 ~ 5.0 にスケーリング

    return float(quality_score)


def _mosnet_spectral_fallback(audio_path: Path) -> float:
    """MOSNetライブラリが利用不可の場合のスペクトル特徴量ベースのMOS推定。

    NISQAフォールバックと同様のアプローチだが、
    MOSスケール (1.0 ~ 5.0) に合わせた異なる重み付けを使用する。

    Args:
        audio_path: 評価対象の音声ファイルのパス。

    Returns:
        推定MOSスコア (1.0 ~ 5.0)。精度は限定的。
    """
    logger.warning(
        "MOSNetフォールバック: スペクトル特徴量ベースの簡易推定を使用します。"
        "結果は参考値です。"
    )

    waveform = _load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)
    audio = waveform.squeeze(0)  # shape: (T,)

    # メルスペクトログラム計算
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(audio)  # shape: (n_mels, time_frames)
    log_mel = torch.log(mel_spec + 1e-10)

    # メルバンドのエネルギー分布: 音声らしいスペクトル形状かどうか
    mel_mean = log_mel.mean(dim=1)  # shape: (n_mels,)
    mel_std = log_mel.std(dim=1)

    # 高周波数帯のエネルギー減衰（自然な音声は高域が減衰する）
    low_band_energy = mel_mean[: n_mels // 3].mean().item()
    high_band_energy = mel_mean[2 * n_mels // 3 :].mean().item()
    spectral_tilt = low_band_energy - high_band_energy

    # フレーム間の変動（自然な音声は適度な変動がある）
    temporal_variation = mel_std.mean().item()

    # RMSエネルギー（極端に小さい/大きい場合は低品質）
    rms = audio.pow(2).mean().sqrt().item()
    rms_score = min(max((rms - 0.005) / 0.1, 0.0), 1.0)

    # ヒューリスティックなスコアマッピング
    tilt_score = min(max(spectral_tilt / 10.0, 0.0), 1.0)
    variation_score = min(max(temporal_variation / 5.0, 0.0), 1.0)

    raw_score = 0.4 * tilt_score + 0.35 * variation_score + 0.25 * rms_score
    mos_score = 1.0 + raw_score * 4.0  # 1.0 ~ 5.0 にスケーリング

    return float(mos_score)


# ---------------------------------------------------------------------------
# NISQA / MOSNet 計算
# ---------------------------------------------------------------------------


def compute_nisqa(
    audio_path: Path,
    nisqa_model: object | None = None,
) -> float:
    """NISQAモデルで知覚品質スコアを計算する。

    nisqaライブラリが利用可能な場合はそれを使用し、
    利用不可の場合はスペクトル特徴量ベースのフォールバック推定を行う。

    Args:
        audio_path: 評価対象の音声ファイルのパス。
        nisqa_model: NISQAモデル。Noneの場合はロードを試みるかフォールバックを使用。

    Returns:
        NISQAスコア。高いほど良い。
        ライブラリ利用不可の場合はフォールバック推定値。
    """
    # nisqa_modelが渡された場合はそれを使用
    if nisqa_model is not None:
        try:
            score = nisqa_model.predict(str(audio_path))
            # predictの戻り値がdictの場合とfloatの場合を考慮
            if isinstance(score, dict):
                return float(score.get("mos_pred", score.get("mos", float("nan"))))
            return float(score)
        except Exception as e:
            logger.warning(f"NISQAモデルでの予測に失敗しました: {e}")
            return _nisqa_spectral_fallback(audio_path)

    # nisqa_modelがNoneの場合、ロードを試みる
    loaded_model = load_nisqa_model(device="cpu")
    if loaded_model is not None:
        try:
            score = loaded_model.predict(str(audio_path))
            if isinstance(score, dict):
                return float(score.get("mos_pred", score.get("mos", float("nan"))))
            return float(score)
        except Exception as e:
            logger.warning(f"NISQAモデルでの予測に失敗しました: {e}")
            return _nisqa_spectral_fallback(audio_path)

    # フォールバック
    return _nisqa_spectral_fallback(audio_path)


def compute_mosnet(
    audio_path: Path,
    mosnet_model: object | None = None,
) -> float:
    """MOSNetモデルで平均オピニオンスコアを推定する。

    mosnetライブラリが利用可能な場合はそれを使用し、
    利用不可の場合はスペクトル特徴量ベースのフォールバック推定を行う。

    Args:
        audio_path: 評価対象の音声ファイルのパス。
        mosnet_model: MOSNetモデル。Noneの場合はロードを試みるかフォールバックを使用。

    Returns:
        推定MOSスコア (1.0 ~ 5.0)。高いほど良い。
        ライブラリ利用不可の場合はフォールバック推定値。
    """
    # mosnet_modelが渡された場合はそれを使用
    if mosnet_model is not None:
        try:
            score = mosnet_model.predict(str(audio_path))
            if isinstance(score, dict):
                return float(score.get("mos", score.get("score", float("nan"))))
            return float(score)
        except Exception as e:
            logger.warning(f"MOSNetモデルでの予測に失敗しました: {e}")
            return _mosnet_spectral_fallback(audio_path)

    # mosnet_modelがNoneの場合、ロードを試みる
    loaded_model = load_mosnet_model(device="cpu")
    if loaded_model is not None:
        try:
            score = loaded_model.predict(str(audio_path))
            if isinstance(score, dict):
                return float(score.get("mos", score.get("score", float("nan"))))
            return float(score)
        except Exception as e:
            logger.warning(f"MOSNetモデルでの予測に失敗しました: {e}")
            return _mosnet_spectral_fallback(audio_path)

    # フォールバック
    return _mosnet_spectral_fallback(audio_path)
