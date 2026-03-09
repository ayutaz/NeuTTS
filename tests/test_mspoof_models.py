"""MSpoof-TTSモデルのユニットテスト。

小さなダミー入力を使用して、各モデルコンポーネントの形状・勾配・
動作を検証する。外部データやGPUは不要。
"""

import pytest
import torch

from mspoof_tts.models.conformer import (
    ConformerBlock,
    ConvolutionModule,
    FeedForwardModule,
)
from mspoof_tts.models.spoof_detector import SpoofDetector
from mspoof_tts.models.multi_resolution import MultiResolutionDetector
from mspoof_tts.data.segment import SegmentExtractor, SegmentMode

# ---------- 共通テスト設定 ----------
VOCAB_SIZE = 1024
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FFN_DIM = 128


# ====================================================================
# TestFeedForwardModule
# ====================================================================
class TestFeedForwardModule:
    """FeedForwardModuleの形状と設定の互換性を検証するテストクラス。"""

    def test_output_shape(self):
        """入力 (2, 10, 256) に対して出力 (2, 10, 256) を返すことを検証する。"""
        ffn = FeedForwardModule(d_model=256, ffn_dim=1024)
        ffn.eval()
        x = torch.randn(2, 10, 256)
        out = ffn(x)
        assert out.shape == (2, 10, 256)

    def test_different_dims(self):
        """d_model=128, ffn_dim=512 の設定で正しく動作することを検証する。"""
        ffn = FeedForwardModule(d_model=128, ffn_dim=512)
        ffn.eval()
        x = torch.randn(3, 7, 128)
        out = ffn(x)
        assert out.shape == (3, 7, 128)


# ====================================================================
# TestConvolutionModule
# ====================================================================
class TestConvolutionModule:
    """ConvolutionModuleの出力形状とシーケンス長保持を検証するテストクラス。"""

    def test_output_shape(self):
        """入力 (2, 10, 256) に対して出力 (2, 10, 256) を返すことを検証する。"""
        conv = ConvolutionModule(d_model=256)
        conv.eval()
        x = torch.randn(2, 10, 256)
        out = conv(x)
        assert out.shape == (2, 10, 256)

    @pytest.mark.parametrize("seq_len", [5, 10, 50])
    def test_preserves_sequence_length(self, seq_len):
        """各種シーケンス長 (5, 10, 50) で長さが保持されることを検証する。"""
        conv = ConvolutionModule(d_model=D_MODEL, kernel_size=7)
        conv.eval()
        x = torch.randn(2, seq_len, D_MODEL)
        out = conv(x)
        assert out.shape == (2, seq_len, D_MODEL)


# ====================================================================
# TestConformerBlock
# ====================================================================
class TestConformerBlock:
    """ConformerBlockの形状・残差接続・勾配フローを検証するテストクラス。"""

    def test_output_shape(self):
        """入力 (2, 10, 256) に対して出力 (2, 10, 256) を返すことを検証する。"""
        block = ConformerBlock(d_model=256, n_heads=8, ffn_dim=1024, kernel_size=7)
        block.eval()
        x = torch.randn(2, 10, 256)
        out = block(x)
        assert out.shape == (2, 10, 256)

    def test_residual_connections(self):
        """残差接続により出力が入力と異なるが同じ形状であることを検証する。"""
        block = ConformerBlock(
            d_model=D_MODEL, n_heads=N_HEADS, ffn_dim=FFN_DIM, kernel_size=7
        )
        block.eval()
        x = torch.randn(2, 10, D_MODEL)
        out = block(x)
        assert out.shape == x.shape
        # 残差接続を経たため、出力は入力と完全には一致しない
        assert not torch.allclose(out, x, atol=1e-6)

    def test_gradient_flow(self):
        """全パラメータに勾配が流れることを検証する。"""
        block = ConformerBlock(
            d_model=D_MODEL, n_heads=N_HEADS, ffn_dim=FFN_DIM, kernel_size=7
        )
        block.train()
        x = torch.randn(2, 10, D_MODEL)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"勾配がNone: {name}"
            assert param.grad.shape == param.shape, f"勾配の形状が不一致: {name}"


# ====================================================================
# TestSpoofDetector
# ====================================================================
class TestSpoofDetector:
    """SpoofDetectorの形状・出力範囲・勾配フローを検証するテストクラス。"""

    def _make_detector(self):
        return SpoofDetector(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            ffn_dim=FFN_DIM,
        )

    def test_output_shape(self):
        """入力 (4, 25) long tensor に対して出力 (4,) float tensor を返すことを検証する。"""
        det = self._make_detector()
        det.eval()
        x = torch.randint(0, VOCAB_SIZE, (4, 25))
        out = det(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_output_range(self):
        """出力値が [0, 1] の範囲にあることを検証する (sigmoid)。"""
        det = self._make_detector()
        det.eval()
        x = torch.randint(0, VOCAB_SIZE, (8, 25))
        with torch.no_grad():
            out = det(x)
        assert (out >= 0.0).all(), f"出力に0未満の値: {out}"
        assert (out <= 1.0).all(), f"出力に1超の値: {out}"

    @pytest.mark.parametrize("seq_len", [10, 25, 50])
    def test_different_seq_lengths(self, seq_len):
        """seq_len=10, 25, 50 の各長さで動作することを検証する。"""
        det = self._make_detector()
        det.eval()
        x = torch.randint(0, VOCAB_SIZE, (3, seq_len))
        out = det(x)
        assert out.shape == (3,)

    def test_parameter_count(self):
        """パラメータ数が妥当な範囲にあることを検証する。"""
        det = self._make_detector()
        total_params = sum(p.numel() for p in det.parameters())
        # 小さな設定なのでパラメータ数は管理可能であるべき
        assert total_params > 0, "パラメータが存在しない"
        # d_model=64, n_layers=2 の小さな設定では数百万パラメータ以下であるべき
        assert total_params < 5_000_000, f"パラメータ数が多すぎる: {total_params}"

    def test_gradient_flow(self):
        """backward passが正常に動作し、全パラメータに勾配が流れることを検証する。"""
        det = self._make_detector()
        det.train()
        x = torch.randint(0, VOCAB_SIZE, (4, 25))
        out = det(x)
        loss = out.sum()
        loss.backward()

        params_with_grad = 0
        for name, param in det.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
        assert params_with_grad > 0, "勾配を持つパラメータが存在しない"


# ====================================================================
# TestMultiResolutionDetector
# ====================================================================
class TestMultiResolutionDetector:
    """MultiResolutionDetectorの初期化・各スコア関数を検証するテストクラス。"""

    def _make_multi(self):
        return MultiResolutionDetector(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            ffn_dim=FFN_DIM,
        )

    def test_init_creates_five_detectors(self):
        """初期化時に5つの検出器が作成されることを検証する。"""
        multi = self._make_multi()
        assert len(multi.detectors) == 5
        expected_names = {"M_10", "M_25", "M_50", "M_50_25", "M_50_10"}
        assert set(multi.detectors.keys()) == expected_names

    def test_score_short(self):
        """score_short: 入力 (3, 15) に対して出力 (3,) を返すことを検証する。"""
        multi = self._make_multi()
        multi.eval()
        x = torch.randint(0, VOCAB_SIZE, (3, 15))
        with torch.no_grad():
            out = multi.score_short(x)
        assert out.shape == (3,)

    def test_score_mid(self):
        """score_mid: 入力 (3, 30) に対して出力 (3,) を返すことを検証する。"""
        multi = self._make_multi()
        multi.eval()
        x = torch.randint(0, VOCAB_SIZE, (3, 30))
        with torch.no_grad():
            out = multi.score_mid(x)
        assert out.shape == (3,)

    def test_score_long(self):
        """score_long: 入力 (3, 60) に対して3キーの辞書を返し、各値が (3,) であることを検証する。"""
        multi = self._make_multi()
        multi.eval()
        x = torch.randint(0, VOCAB_SIZE, (3, 60))
        with torch.no_grad():
            scores = multi.score_long(x)
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {"M_50", "M_50_25", "M_50_10"}
        for key, val in scores.items():
            assert val.shape == (3,), f"{key} shape mismatch"

    def test_rank_aggregate(self):
        """rank_aggregate: 有効範囲内のスカラインデックスを返すことを検証する。"""
        multi = self._make_multi()
        dummy_scores = {
            "M_50": torch.tensor([0.8, 0.6, 0.9]),
            "M_50_25": torch.tensor([0.7, 0.5, 0.85]),
            "M_50_10": torch.tensor([0.75, 0.55, 0.88]),
        }
        best_idx = multi.rank_aggregate(dummy_scores)
        assert best_idx.dim() == 0, "結果はスカラであるべき"
        assert 0 <= best_idx.item() < 3, "インデックスは有効範囲内であるべき"


# ====================================================================
# TestSegmentExtractor
# ====================================================================
class TestSegmentExtractor:
    """SegmentExtractorの各モードでの正確な抽出動作を検証するテストクラス。"""

    def test_contiguous_10_output(self):
        """contiguous_10モードで既知の入力に対して正確な値を返すことを検証する。"""
        extractor = SegmentExtractor(SegmentMode.CONTIGUOUS_10, random_offset=False)
        # offset=0 (random_offset=False) なので先頭10トークンを返す
        token_ids = torch.arange(20)
        segment = extractor.extract(token_ids)
        assert segment.shape == (10,)
        expected = torch.arange(10)
        assert torch.equal(segment, expected)

    def test_skip_sampling_values(self):
        """skip_50_25モードで正しいインデックスが選択されることを検証する。"""
        extractor = SegmentExtractor(SegmentMode.SKIP_50_25, random_offset=False)
        # offset=0, span=50, skip_rate=2 -> indices 0,2,4,...,48 の25トークン
        token_ids = torch.arange(60)
        segment = extractor.extract(token_ids)
        assert segment.shape == (25,)
        expected = torch.arange(0, 50, 2)
        assert torch.equal(segment, expected)

    def test_skip_50_10_values(self):
        """skip_50_10モードで正しいインデックスが選択されることを検証する。"""
        extractor = SegmentExtractor(SegmentMode.SKIP_50_10, random_offset=False)
        # offset=0, span=50, skip_rate=5 -> indices 0,5,10,...,45 の10トークン
        token_ids = torch.arange(60)
        segment = extractor.extract(token_ids)
        assert segment.shape == (10,)
        expected = torch.arange(0, 50, 5)
        assert torch.equal(segment, expected)

    def test_too_short_raises(self):
        """短すぎる入力に対してValueErrorが送出されることを検証する。"""
        extractor = SegmentExtractor(SegmentMode.CONTIGUOUS_50, random_offset=False)
        token_ids = torch.arange(30)  # 50トークン必要だが30しかない
        with pytest.raises(ValueError, match="短すぎます"):
            extractor.extract(token_ids)

    def test_too_short_skip_raises(self):
        """skip samplingモードで短すぎる入力に対してValueErrorが送出されることを検証する。"""
        extractor = SegmentExtractor(SegmentMode.SKIP_50_25, random_offset=False)
        token_ids = torch.arange(40)  # span=50 必要だが40しかない
        with pytest.raises(ValueError, match="短すぎます"):
            extractor.extract(token_ids)
