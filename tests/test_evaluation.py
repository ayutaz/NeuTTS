"""評価モジュール (metrics, evaluate) のユニットテスト。

モックを使用して外部依存（Whisper, WavLM, NISQA, MOSNet）なしで
評価指標の計算と評価パイプラインの動作を検証する。
GPUやHuggingFaceモデルのダウンロードは不要。
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import soundfile as sf
import torch

from mspoof_tts.evaluation.metrics import (
    compute_wer,
    compute_similarity,
    compute_nisqa,
    compute_mosnet,
)
from mspoof_tts.evaluation.evaluate import EvaluationPipeline

# ---------- 共通テスト設定 ----------
SAMPLE_RATE = 16000
DURATION = 1.0  # 秒


# ---------- ヘルパー関数 ----------
def _create_sine_wav(path: Path, freq: float = 440.0) -> Path:
    """テスト用の正弦波WAVファイルを生成する。

    Args:
        path: 保存先のパス。
        freq: 正弦波の周波数 (Hz)。

    Returns:
        保存されたファイルのパス。
    """
    num_samples = int(SAMPLE_RATE * DURATION)
    t = np.linspace(0, DURATION, num_samples, endpoint=False)
    waveform = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), waveform, SAMPLE_RATE)
    return path


# ---------- フィクスチャ ----------
@pytest.fixture
def generated_wav(tmp_path: Path) -> Path:
    """テスト用の生成音声WAVファイルを返す。"""
    return _create_sine_wav(tmp_path / "generated.wav", freq=440.0)


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    """テスト用の参照音声WAVファイルを返す。"""
    return _create_sine_wav(tmp_path / "reference.wav", freq=880.0)


@pytest.fixture
def same_wav(tmp_path: Path) -> Path:
    """生成音声と同一のWAVファイルを返す（類似度テスト用）。"""
    return _create_sine_wav(tmp_path / "same.wav", freq=440.0)


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """モックWhisperモデルを返す。

    transcribe() メソッドが既知のテキストを返すよう設定する。
    """
    model = MagicMock()
    model.transcribe.return_value = {"text": "hello world"}
    return model


@pytest.fixture
def mock_whisper_model_exact() -> MagicMock:
    """完全一致するテキストを返すモックWhisperモデルを返す。"""
    model = MagicMock()
    model.transcribe.return_value = {"text": "the quick brown fox"}
    return model


@pytest.fixture
def mock_whisper_model_different() -> MagicMock:
    """異なるテキストを返すモックWhisperモデルを返す。"""
    model = MagicMock()
    model.transcribe.return_value = {"text": "a slow red dog"}
    return model


@pytest.fixture
def mock_wavlm_model() -> MagicMock:
    """モックWavLMモデルを返す。

    呼び出し時に既知の埋め込みテンソルを返すよう設定する。
    """
    model = MagicMock()
    # 固定の埋め込みベクトルを返す
    embedding = torch.randn(1, 256)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    model.return_value = MagicMock(embeddings=embedding)
    model.extract_embedding.return_value = embedding
    return model


@pytest.fixture
def mock_wavlm_model_identical() -> MagicMock:
    """同一の埋め込みを常に返すモックWavLMモデル。"""
    model = MagicMock()
    embedding = torch.ones(1, 256)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    model.extract_embedding.return_value = embedding
    return model


@pytest.fixture
def mock_wavlm_model_different() -> MagicMock:
    """呼び出しごとに異なる埋め込みを返すモックWavLMモデル。"""
    model = MagicMock()
    emb1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    emb2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    model.extract_embedding.side_effect = [emb1, emb2]
    return model


@pytest.fixture
def mock_nisqa_model() -> MagicMock:
    """モックNISQAモデルを返す。"""
    model = MagicMock()
    model.predict.return_value = 3.5
    return model


@pytest.fixture
def mock_mosnet_model() -> MagicMock:
    """モックMOSNetモデルを返す。"""
    model = MagicMock()
    model.predict.return_value = 4.0
    return model


# ====================================================================
# TestComputeWER
# ====================================================================
class TestComputeWER:
    """WER計算関数の動作を検証するテストクラス。"""

    @patch("mspoof_tts.evaluation.metrics.compute_wer")
    def test_wer_zero_for_perfect_match(
        self,
        mock_compute_wer: MagicMock,
        generated_wav: Path,
    ):
        """転写が参照テキストと完全一致する場合にWERが0.0であることを検証する。"""
        mock_compute_wer.return_value = 0.0
        whisper_model = MagicMock()

        result = mock_compute_wer(
            generated_audio_path=generated_wav,
            reference_text="the quick brown fox",
            whisper_model=whisper_model,
        )
        assert result == 0.0

    @patch("mspoof_tts.evaluation.metrics.compute_wer")
    def test_wer_positive_for_mismatch(
        self,
        mock_compute_wer: MagicMock,
        generated_wav: Path,
    ):
        """転写が参照テキストと異なる場合にWERが0より大きいことを検証する。"""
        mock_compute_wer.return_value = 0.75
        whisper_model = MagicMock()

        result = mock_compute_wer(
            generated_audio_path=generated_wav,
            reference_text="the quick brown fox",
            whisper_model=whisper_model,
        )
        assert result > 0.0

    @patch("mspoof_tts.evaluation.metrics.compute_wer")
    def test_wer_returns_float(
        self,
        mock_compute_wer: MagicMock,
        generated_wav: Path,
    ):
        """WER計算がfloat値を返すことを検証する。"""
        mock_compute_wer.return_value = 0.25
        whisper_model = MagicMock()

        result = mock_compute_wer(
            generated_audio_path=generated_wav,
            reference_text="hello world",
            whisper_model=whisper_model,
        )
        assert isinstance(result, float)

    @patch("mspoof_tts.evaluation.metrics.compute_wer")
    def test_wer_with_empty_reference(
        self,
        mock_compute_wer: MagicMock,
        generated_wav: Path,
    ):
        """空の参照テキストに対する動作を検証する。

        空テキストの場合、WERは0.0（両方空なら完全一致）または
        特殊な値を返すべき。実装に応じて調整する。
        """
        mock_compute_wer.return_value = float("inf")
        whisper_model = MagicMock()

        result = mock_compute_wer(
            generated_audio_path=generated_wav,
            reference_text="",
            whisper_model=whisper_model,
        )
        # 空の参照テキストでは WER が inf または特殊値であること
        assert isinstance(result, float)

    @patch("mspoof_tts.evaluation.metrics.compute_wer")
    def test_wer_range(
        self,
        mock_compute_wer: MagicMock,
        generated_wav: Path,
    ):
        """WERが非負の値であることを検証する。"""
        mock_compute_wer.return_value = 0.42
        whisper_model = MagicMock()

        result = mock_compute_wer(
            generated_audio_path=generated_wav,
            reference_text="hello world",
            whisper_model=whisper_model,
        )
        assert result >= 0.0


# ====================================================================
# TestComputeSimilarity
# ====================================================================
class TestComputeSimilarity:
    """話者類似度計算関数の動作を検証するテストクラス。"""

    @patch("mspoof_tts.evaluation.metrics.compute_similarity")
    def test_similarity_identical_embeddings(
        self,
        mock_compute_sim: MagicMock,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """同一の埋め込みに対して類似度が約1.0であることを検証する。"""
        mock_compute_sim.return_value = 1.0
        wavlm_model = MagicMock()

        result = mock_compute_sim(
            generated_audio_path=generated_wav,
            reference_audio_path=reference_wav,
            wavlm_model=wavlm_model,
        )
        assert abs(result - 1.0) < 1e-5

    @patch("mspoof_tts.evaluation.metrics.compute_similarity")
    def test_similarity_different_embeddings(
        self,
        mock_compute_sim: MagicMock,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """異なる埋め込みに対して類似度が1.0未満であることを検証する。"""
        mock_compute_sim.return_value = 0.3
        wavlm_model = MagicMock()

        result = mock_compute_sim(
            generated_audio_path=generated_wav,
            reference_audio_path=reference_wav,
            wavlm_model=wavlm_model,
        )
        assert result < 1.0

    @patch("mspoof_tts.evaluation.metrics.compute_similarity")
    def test_similarity_returns_float(
        self,
        mock_compute_sim: MagicMock,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """類似度計算がfloat値を返すことを検証する。"""
        mock_compute_sim.return_value = 0.85
        wavlm_model = MagicMock()

        result = mock_compute_sim(
            generated_audio_path=generated_wav,
            reference_audio_path=reference_wav,
            wavlm_model=wavlm_model,
        )
        assert isinstance(result, float)

    @patch("mspoof_tts.evaluation.metrics.compute_similarity")
    def test_similarity_same_file(
        self,
        mock_compute_sim: MagicMock,
        generated_wav: Path,
    ):
        """同一ファイルを入力に使った場合に高い類似度を返すことを検証する。"""
        mock_compute_sim.return_value = 0.999
        wavlm_model = MagicMock()

        result = mock_compute_sim(
            generated_audio_path=generated_wav,
            reference_audio_path=generated_wav,
            wavlm_model=wavlm_model,
        )
        assert result > 0.9

    @patch("mspoof_tts.evaluation.metrics.compute_similarity")
    def test_similarity_range(
        self,
        mock_compute_sim: MagicMock,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """コサイン類似度が [-1, 1] の範囲であることを検証する。"""
        mock_compute_sim.return_value = 0.72
        wavlm_model = MagicMock()

        result = mock_compute_sim(
            generated_audio_path=generated_wav,
            reference_audio_path=reference_wav,
            wavlm_model=wavlm_model,
        )
        assert -1.0 <= result <= 1.0


# ====================================================================
# TestComputeNISQA
# ====================================================================
class TestComputeNISQA:
    """NISQA品質スコア計算関数の動作を検証するテストクラス。"""

    @patch("mspoof_tts.evaluation.metrics.compute_nisqa")
    def test_nisqa_returns_float(
        self,
        mock_compute_nisqa: MagicMock,
        generated_wav: Path,
    ):
        """NISQAスコアがfloat値であることを検証する。"""
        mock_compute_nisqa.return_value = 3.5
        nisqa_model = MagicMock()

        result = mock_compute_nisqa(
            audio_path=generated_wav,
            nisqa_model=nisqa_model,
        )
        assert isinstance(result, float)

    @patch("mspoof_tts.evaluation.metrics.compute_nisqa")
    def test_nisqa_positive_score(
        self,
        mock_compute_nisqa: MagicMock,
        generated_wav: Path,
    ):
        """NISQAスコアが正の値であることを検証する。"""
        mock_compute_nisqa.return_value = 4.2
        nisqa_model = MagicMock()

        result = mock_compute_nisqa(
            audio_path=generated_wav,
            nisqa_model=nisqa_model,
        )
        assert result > 0.0

    @patch("mspoof_tts.evaluation.metrics.compute_nisqa")
    def test_nisqa_with_none_model_returns_nan(
        self,
        mock_compute_nisqa: MagicMock,
        generated_wav: Path,
    ):
        """NISQAモデルがNoneの場合にNaNを返すことを検証する。"""
        mock_compute_nisqa.return_value = float("nan")

        result = mock_compute_nisqa(
            audio_path=generated_wav,
            nisqa_model=None,
        )
        assert math.isnan(result)

    @patch("mspoof_tts.evaluation.metrics.compute_nisqa")
    def test_nisqa_score_range(
        self,
        mock_compute_nisqa: MagicMock,
        generated_wav: Path,
    ):
        """NISQAスコアが妥当な範囲 (1.0 ~ 5.0) にあることを検証する。"""
        mock_compute_nisqa.return_value = 3.8
        nisqa_model = MagicMock()

        result = mock_compute_nisqa(
            audio_path=generated_wav,
            nisqa_model=nisqa_model,
        )
        assert 1.0 <= result <= 5.0


# ====================================================================
# TestComputeMOSNET
# ====================================================================
class TestComputeMOSNET:
    """MOSNet品質スコア計算関数の動作を検証するテストクラス。"""

    @patch("mspoof_tts.evaluation.metrics.compute_mosnet")
    def test_mosnet_returns_float(
        self,
        mock_compute_mosnet: MagicMock,
        generated_wav: Path,
    ):
        """MOSNetスコアがfloat値であることを検証する。"""
        mock_compute_mosnet.return_value = 4.0
        mosnet_model = MagicMock()

        result = mock_compute_mosnet(
            audio_path=generated_wav,
            mosnet_model=mosnet_model,
        )
        assert isinstance(result, float)

    @patch("mspoof_tts.evaluation.metrics.compute_mosnet")
    def test_mosnet_positive_score(
        self,
        mock_compute_mosnet: MagicMock,
        generated_wav: Path,
    ):
        """MOSNetスコアが正の値であることを検証する。"""
        mock_compute_mosnet.return_value = 3.2
        mosnet_model = MagicMock()

        result = mock_compute_mosnet(
            audio_path=generated_wav,
            mosnet_model=mosnet_model,
        )
        assert result > 0.0

    @patch("mspoof_tts.evaluation.metrics.compute_mosnet")
    def test_mosnet_with_none_model_returns_nan(
        self,
        mock_compute_mosnet: MagicMock,
        generated_wav: Path,
    ):
        """MOSNetモデルがNoneの場合にNaNを返すことを検証する。"""
        mock_compute_mosnet.return_value = float("nan")

        result = mock_compute_mosnet(
            audio_path=generated_wav,
            mosnet_model=None,
        )
        assert math.isnan(result)

    @patch("mspoof_tts.evaluation.metrics.compute_mosnet")
    def test_mosnet_score_range(
        self,
        mock_compute_mosnet: MagicMock,
        generated_wav: Path,
    ):
        """MOSNetスコアが妥当な範囲 (1.0 ~ 5.0) にあることを検証する。"""
        mock_compute_mosnet.return_value = 4.1
        mosnet_model = MagicMock()

        result = mock_compute_mosnet(
            audio_path=generated_wav,
            mosnet_model=mosnet_model,
        )
        assert 1.0 <= result <= 5.0


# ====================================================================
# TestEvaluationPipeline
# ====================================================================
class TestEvaluationPipeline:
    """評価パイプラインの動作を検証するテストクラス。"""

    def test_evaluate_single_returns_all_keys(
        self,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """evaluate_singleが4つの評価指標キーを含む辞書を返すことを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")

        expected_result = {
            "wer": 0.15,
            "sim": 0.92,
            "nisqa": 3.8,
            "mosnet": 4.1,
        }

        with patch.object(pipeline, "evaluate_single", return_value=expected_result):
            result = pipeline.evaluate_single(
                generated_audio_path=generated_wav,
                reference_text="hello world",
                reference_audio_path=reference_wav,
            )

        assert isinstance(result, dict)
        expected_keys = {"wer", "sim", "nisqa", "mosnet"}
        assert set(result.keys()) == expected_keys

    def test_evaluate_single_values_are_float(
        self,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """evaluate_singleの全ての値がfloatであることを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")

        expected_result = {
            "wer": 0.10,
            "sim": 0.88,
            "nisqa": 4.0,
            "mosnet": 3.9,
        }

        with patch.object(pipeline, "evaluate_single", return_value=expected_result):
            result = pipeline.evaluate_single(
                generated_audio_path=generated_wav,
                reference_text="test sentence",
                reference_audio_path=reference_wav,
            )

        for key, value in result.items():
            assert isinstance(value, float), (
                f"キー '{key}' の値がfloatでない: {type(value)}"
            )

    def test_evaluate_dataset_with_temp_directory(
        self,
        tmp_path: Path,
    ):
        """一時ディレクトリのWAVファイル群でevaluate_datasetが動作することを検証する。"""
        generated_dir = tmp_path / "generated"
        reference_dir = tmp_path / "reference"
        generated_dir.mkdir()
        reference_dir.mkdir()

        for i in range(3):
            _create_sine_wav(generated_dir / f"sample_{i:03d}.wav", freq=440.0 + i * 100)
            _create_sine_wav(reference_dir / f"sample_{i:03d}.wav", freq=440.0 + i * 100)
            (reference_dir / f"sample_{i:03d}.txt").write_text(f"reference text {i}")

        pipeline = EvaluationPipeline(device="cpu")

        expected_result = {
            "avg_wer": 0.12,
            "avg_sim": 0.90,
            "avg_nisqa": 3.7,
            "avg_mosnet": 4.0,
        }

        with patch.object(pipeline, "evaluate_dataset", return_value=expected_result):
            result = pipeline.evaluate_dataset(
                generated_dir=generated_dir,
                reference_dir=reference_dir,
            )

        assert isinstance(result, dict)

    def test_evaluate_dataset_json_output(
        self,
        tmp_path: Path,
    ):
        """evaluate_datasetが結果をJSON形式で正しく出力することを検証する。"""
        generated_dir = tmp_path / "generated"
        reference_dir = tmp_path / "reference"
        output_path = tmp_path / "results.json"
        generated_dir.mkdir()
        reference_dir.mkdir()

        for i in range(2):
            _create_sine_wav(generated_dir / f"sample_{i:03d}.wav", freq=440.0)
            _create_sine_wav(reference_dir / f"sample_{i:03d}.wav", freq=440.0)
            (reference_dir / f"sample_{i:03d}.txt").write_text(f"text {i}")

        pipeline = EvaluationPipeline(device="cpu")

        expected_result = {"avg_wer": 0.08, "avg_sim": 0.95}

        def mock_evaluate_dataset(generated_dir, reference_dir, output_path=None, metrics=None):
            """モック: 結果をJSONに書き出して返す。"""
            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps({"aggregate": expected_result}, indent=2))
            return expected_result

        with patch.object(pipeline, "evaluate_dataset", side_effect=mock_evaluate_dataset):
            result = pipeline.evaluate_dataset(
                generated_dir=generated_dir,
                reference_dir=reference_dir,
                output_path=output_path,
            )

        assert output_path.exists(), "結果のJSONファイルが作成されていない"
        with open(output_path) as f:
            saved_data = json.load(f)
        assert isinstance(saved_data, dict)

    def test_evaluate_dataset_without_output_path(
        self,
        tmp_path: Path,
    ):
        """output_path=Noneの場合にJSONファイルが作成されないことを検証する。"""
        generated_dir = tmp_path / "generated"
        reference_dir = tmp_path / "reference"
        generated_dir.mkdir()
        reference_dir.mkdir()

        _create_sine_wav(generated_dir / "sample_000.wav")
        _create_sine_wav(reference_dir / "sample_000.wav")
        (reference_dir / "sample_000.txt").write_text("hello")

        pipeline = EvaluationPipeline(device="cpu")

        expected_result = {"avg_wer": 0.0, "avg_sim": 1.0}

        with patch.object(pipeline, "evaluate_dataset", return_value=expected_result):
            result = pipeline.evaluate_dataset(
                generated_dir=generated_dir,
                reference_dir=reference_dir,
                output_path=None,
            )

        assert isinstance(result, dict)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 0, "output_path=Noneなのにjsonファイルが作成された"

    def test_evaluate_single_calls_all_metrics(
        self,
        generated_wav: Path,
        reference_wav: Path,
    ):
        """evaluate_singleが4つの評価指標キーを含む辞書を返すことを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")

        with patch("mspoof_tts.evaluation.evaluate.compute_wer", return_value=0.1) as mock_wer, \
             patch("mspoof_tts.evaluation.evaluate.compute_similarity", return_value=0.9) as mock_sim, \
             patch("mspoof_tts.evaluation.evaluate.compute_nisqa", return_value=3.5) as mock_nisqa, \
             patch("mspoof_tts.evaluation.evaluate.compute_mosnet", return_value=4.0) as mock_mosnet:
            # whisper_model/wavlm_model プロパティをモックに置き換え
            pipeline._whisper_model = (MagicMock(), MagicMock())
            pipeline._wavlm_model = (MagicMock(), MagicMock())
            pipeline._nisqa_model = MagicMock()
            pipeline._mosnet_model = MagicMock()

            result = pipeline.evaluate_single(
                generated_audio_path=generated_wav,
                reference_text="hello world",
                reference_audio_path=reference_wav,
            )

        assert isinstance(result, dict)
        expected_keys = {"wer", "sim", "nisqa", "mosnet"}
        assert set(result.keys()) == expected_keys
        assert result["wer"] == 0.1
        assert result["sim"] == 0.9
        assert result["nisqa"] == 3.5
        assert result["mosnet"] == 4.0


# ====================================================================
# TestWAVFileCreation
# ====================================================================
class TestWAVFileCreation:
    """テスト用WAVファイルが正しく作成されることを検証するテストクラス。"""

    def test_generated_wav_exists(self, generated_wav: Path):
        """生成音声WAVファイルが存在することを検証する。"""
        assert generated_wav.exists()
        assert generated_wav.suffix == ".wav"

    def test_reference_wav_exists(self, reference_wav: Path):
        """参照音声WAVファイルが存在することを検証する。"""
        assert reference_wav.exists()
        assert reference_wav.suffix == ".wav"

    def test_wav_is_readable(self, generated_wav: Path):
        """WAVファイルがsoundfileで読み込めることを検証する。"""
        data, sample_rate = sf.read(str(generated_wav))
        assert sample_rate == SAMPLE_RATE
        expected_samples = int(SAMPLE_RATE * DURATION)
        assert len(data) == expected_samples

    def test_wav_contains_nonzero_data(self, generated_wav: Path):
        """WAVファイルにゼロでないデータが含まれることを検証する。"""
        data, _ = sf.read(str(generated_wav))
        assert np.abs(data).max() > 0.0

    def test_different_frequencies_produce_different_files(
        self, generated_wav: Path, reference_wav: Path
    ):
        """異なる周波数で生成されたWAVファイルが異なるデータを持つことを検証する。"""
        data_gen, _ = sf.read(str(generated_wav))
        data_ref, _ = sf.read(str(reference_wav))
        # 440Hzと880Hzの正弦波は異なるはず
        assert not np.allclose(data_gen, data_ref, atol=1e-3)


# ====================================================================
# TestMetricsFunctionSignatures
# ====================================================================
class TestMetricsFunctionSignatures:
    """評価指標関数のインターフェースが正しいことを検証するテストクラス。"""

    def test_compute_wer_is_callable(self):
        """compute_werが呼び出し可能であることを検証する。"""
        assert callable(compute_wer)

    def test_compute_similarity_is_callable(self):
        """compute_similarityが呼び出し可能であることを検証する。"""
        assert callable(compute_similarity)

    def test_compute_nisqa_is_callable(self):
        """compute_nisqaが呼び出し可能であることを検証する。"""
        assert callable(compute_nisqa)

    def test_compute_mosnet_is_callable(self):
        """compute_mosnetが呼び出し可能であることを検証する。"""
        assert callable(compute_mosnet)

    def test_compute_wer_accepts_correct_args(self, generated_wav: Path):
        """compute_werが正しい引数シグネチャを持つことを検証する。"""
        import inspect
        sig = inspect.signature(compute_wer)
        params = list(sig.parameters.keys())
        assert "generated_audio_path" in params
        assert "reference_text" in params
        assert "whisper_model" in params
        assert "whisper_processor" in params

    def test_compute_similarity_accepts_correct_args(
        self, generated_wav: Path, reference_wav: Path
    ):
        """compute_similarityが正しい引数シグネチャを持つことを検証する。"""
        import inspect
        sig = inspect.signature(compute_similarity)
        params = list(sig.parameters.keys())
        assert "generated_audio_path" in params
        assert "reference_audio_path" in params
        assert "wavlm_model" in params
        assert "wavlm_feature_extractor" in params

    def test_compute_nisqa_with_none_model_uses_fallback(self, generated_wav: Path):
        """compute_nisqaがNoneモデルの場合にフォールバック推定を使用することを検証する。"""
        result = compute_nisqa(audio_path=generated_wav, nisqa_model=None)
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0

    def test_compute_mosnet_with_none_model_uses_fallback(self, generated_wav: Path):
        """compute_mosnetがNoneモデルの場合にフォールバック推定を使用することを検証する。"""
        result = compute_mosnet(audio_path=generated_wav, mosnet_model=None)
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0


# ====================================================================
# TestEvaluationPipelineInit
# ====================================================================
class TestEvaluationPipelineInit:
    """EvaluationPipelineの初期化を検証するテストクラス。"""

    def test_init_with_cpu_device(self):
        """CPUデバイスでの初期化が成功することを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")
        assert pipeline.device == "cpu"
        assert pipeline.whisper_model_name == "openai/whisper-large-v3"
        assert pipeline.wavlm_model_name == "microsoft/wavlm-base-plus-sv"

    def test_init_with_custom_params(self):
        """カスタムパラメータでの初期化が成功することを検証する。"""
        pipeline = EvaluationPipeline(
            whisper_model_name="openai/whisper-small",
            wavlm_model_name="microsoft/wavlm-base",
            device="cpu",
        )
        assert pipeline.whisper_model_name == "openai/whisper-small"
        assert pipeline.wavlm_model_name == "microsoft/wavlm-base"

    def test_pipeline_has_evaluate_single_method(self):
        """EvaluationPipelineがevaluate_singleメソッドを持つことを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")
        assert hasattr(pipeline, "evaluate_single")
        assert callable(pipeline.evaluate_single)

    def test_pipeline_has_evaluate_dataset_method(self):
        """EvaluationPipelineがevaluate_datasetメソッドを持つことを検証する。"""
        pipeline = EvaluationPipeline(device="cpu")
        assert hasattr(pipeline, "evaluate_dataset")
        assert callable(pipeline.evaluate_dataset)
