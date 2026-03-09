"""階層的デコーダー (HierarchicalDecoder) のユニットテスト。

MockBackboneModelを使用して、階層的デコーダーの各フェーズ
（ウォームアップ、3ステージ枝刈り、最終選択、エンドツーエンド）を検証する。
小さな次元設定で高速に実行する。
"""

import pytest
import torch

from mspoof_tts.models.multi_resolution import MultiResolutionDetector
from mspoof_tts.sampling.eas import EntropyAwareSampler
from mspoof_tts.sampling.hierarchical import HierarchicalDecoder

# ---------- 共通テスト設定 ----------
VOCAB_SIZE = 1024
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FFN_DIM = 128

# 高速テスト用の小さなデコーディング設定
SMALL_WARMUP = 5
SMALL_STAGE_LENGTHS = [3, 5, 8]
SMALL_BEAM_SIZES = [3, 2, 2]
SMALL_MAX_LENGTH = 50


# ---------- モックモデル ----------
class MockOutput:
    """モデル出力を模倣するデータクラス。"""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.past_key_values = None


class MockBackboneModel:
    """テスト用のダミーバックボーンモデル。"""

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> MockOutput:
        if input_ids.dim() == 1:
            batch, seq_len = 1, input_ids.shape[0]
        else:
            batch, seq_len = input_ids.shape
        logits = torch.randn(batch, seq_len, self.vocab_size)
        return MockOutput(logits=logits)


# ---------- フィクスチャ ----------
@pytest.fixture
def mock_model() -> MockBackboneModel:
    """テスト用のダミーバックボーンモデルを返す。"""
    return MockBackboneModel(vocab_size=VOCAB_SIZE)


@pytest.fixture
def eas_sampler() -> EntropyAwareSampler:
    """テスト用のEASサンプラーを返す。"""
    return EntropyAwareSampler(
        top_k=50,
        top_p=0.8,
        temperature=1.0,
        cluster_size=3,
        memory_window=15,
        alpha=0.2,
        beta=0.7,
        gamma=0.8,
    )


@pytest.fixture
def detectors() -> MultiResolutionDetector:
    """テスト用のマルチ解像度検出器を返す（小さな次元設定）。"""
    det = MultiResolutionDetector(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn_dim=FFN_DIM,
    )
    det.eval()
    return det


@pytest.fixture
def small_decoder(
    eas_sampler: EntropyAwareSampler,
    detectors: MultiResolutionDetector,
) -> HierarchicalDecoder:
    """小さな設定の階層的デコーダーを返す。"""
    return HierarchicalDecoder(
        eas_sampler=eas_sampler,
        detectors=detectors,
        warmup=SMALL_WARMUP,
        stage_lengths=SMALL_STAGE_LENGTHS,
        beam_sizes=SMALL_BEAM_SIZES,
        max_length=SMALL_MAX_LENGTH,
    )


@pytest.fixture
def prefix() -> torch.Tensor:
    """テスト用のプレフィックストークン列を返す。"""
    return torch.randint(0, VOCAB_SIZE, (20,))


# ====================================================================
# TestHierarchicalDecoderInit
# ====================================================================
class TestHierarchicalDecoderInit:
    """階層的デコーダーの初期化パラメータを検証するテストクラス。"""

    def test_default_params(
        self,
        eas_sampler: EntropyAwareSampler,
        detectors: MultiResolutionDetector,
    ):
        """デフォルトパラメータ (warmup=20, stage_lengths=[10,25,50], beam_sizes=[8,5,3]) が正しく設定されることを検証する。"""
        decoder = HierarchicalDecoder(
            eas_sampler=eas_sampler,
            detectors=detectors,
        )
        assert decoder.warmup == 20
        assert decoder.stage_lengths == [10, 25, 50]
        assert decoder.beam_sizes == [8, 5, 3]
        assert decoder.rank_weights == [1.0, 1.0, 1.0]
        assert decoder.max_length == 2048

    def test_custom_params(
        self,
        eas_sampler: EntropyAwareSampler,
        detectors: MultiResolutionDetector,
    ):
        """カスタムパラメータが正しく保存されることを検証する。"""
        decoder = HierarchicalDecoder(
            eas_sampler=eas_sampler,
            detectors=detectors,
            warmup=10,
            stage_lengths=[5, 15, 30],
            beam_sizes=[4, 3, 2],
            rank_weights=[0.5, 1.0, 1.5],
            max_length=512,
        )
        assert decoder.warmup == 10
        assert decoder.stage_lengths == [5, 15, 30]
        assert decoder.beam_sizes == [4, 3, 2]
        assert decoder.rank_weights == [0.5, 1.0, 1.5]
        assert decoder.max_length == 512


# ====================================================================
# TestWarmupPhase
# ====================================================================
class TestWarmupPhase:
    """ウォームアップフェーズの動作を検証するテストクラス。"""

    def test_warmup_generates_correct_length(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """ウォームアップがプレフィックスにちょうどL_wトークンを追加することを検証する。"""
        result = small_decoder._warmup_phase(mock_model, prefix)
        expected_length = len(prefix) + SMALL_WARMUP
        assert len(result) == expected_length, (
            f"ウォームアップ後の長さが不正: expected {expected_length}, got {len(result)}"
        )

    def test_warmup_returns_tensor(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """ウォームアップの結果が1Dテンソルであることを検証する。"""
        result = small_decoder._warmup_phase(mock_model, prefix)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1


# ====================================================================
# TestStage1Prune
# ====================================================================
class TestStage1Prune:
    """Stage 1（短スパン枝刈り）の動作を検証するテストクラス。"""

    def test_returns_correct_number_of_candidates(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """Stage 1がB1個の候補を返すことを検証する。"""
        candidates = small_decoder._stage1_prune(mock_model, prefix)
        expected_count = SMALL_BEAM_SIZES[1]  # B1
        assert len(candidates) == expected_count, (
            f"Stage 1の候補数が不正: expected {expected_count}, got {len(candidates)}"
        )

    def test_candidates_are_extended(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """各候補がプレフィックスより長いことを検証する。"""
        candidates = small_decoder._stage1_prune(mock_model, prefix)
        prefix_len = len(prefix)
        for i, item in enumerate(candidates):
            # _stage1_prune は (tensor, sampler) タプルを返す
            cand = item[0] if isinstance(item, tuple) else item
            assert isinstance(cand, torch.Tensor)
            assert len(cand) > prefix_len, (
                f"候補 {i} がプレフィックスより長くない: "
                f"candidate len={len(cand)}, prefix len={prefix_len}"
            )


# ====================================================================
# TestStage2Prune
# ====================================================================
class TestStage2Prune:
    """Stage 2（中距離枝刈り）の動作を検証するテストクラス。"""

    def test_returns_correct_number_of_candidates(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """Stage 2がB2個の候補を返すことを検証する。"""
        # Stage 1の出力を先に取得
        stage1_candidates = small_decoder._stage1_prune(mock_model, prefix)
        candidates = small_decoder._stage2_prune(mock_model, stage1_candidates)
        expected_count = SMALL_BEAM_SIZES[2]  # B2
        assert len(candidates) == expected_count, (
            f"Stage 2の候補数が不正: expected {expected_count}, got {len(candidates)}"
        )

    def test_candidates_are_further_extended(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """Stage 2の候補がStage 1の出力より長いことを検証する。"""
        stage1_candidates = small_decoder._stage1_prune(mock_model, prefix)
        stage1_lengths = [
            len(c[0] if isinstance(c, tuple) else c) for c in stage1_candidates
        ]
        min_stage1_len = min(stage1_lengths)

        candidates = small_decoder._stage2_prune(mock_model, stage1_candidates)
        for i, item in enumerate(candidates):
            cand = item[0] if isinstance(item, tuple) else item
            assert isinstance(cand, torch.Tensor)
            assert len(cand) > min_stage1_len, (
                f"Stage 2候補 {i} がStage 1出力より長くない: "
                f"candidate len={len(cand)}, min stage1 len={min_stage1_len}"
            )


# ====================================================================
# TestStage3Extend
# ====================================================================
class TestStage3Extend:
    """Stage 3（長スパン延長）の動作を検証するテストクラス。"""

    def test_candidates_are_extended_to_L3(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """各候補がL3トークン分だけプレフィックスより長くなることを検証する。"""
        # Stage 1 -> Stage 2 の出力を準備
        stage1_candidates = small_decoder._stage1_prune(mock_model, prefix)
        stage2_candidates = small_decoder._stage2_prune(
            mock_model, stage1_candidates
        )
        stage2_lengths = [len(c) for c in stage2_candidates]

        candidates = small_decoder._stage3_extend(mock_model, stage2_candidates)

        # Stage 3の候補数はStage 2と同じ (B2)
        assert len(candidates) == len(stage2_candidates)

        # 各候補がStage 2の出力より長くなっていること
        for i, cand in enumerate(candidates):
            assert isinstance(cand, torch.Tensor)
            assert len(cand) > stage2_lengths[i], (
                f"Stage 3候補 {i} がStage 2出力より長くない: "
                f"candidate len={len(cand)}, stage2 len={stage2_lengths[i]}"
            )


# ====================================================================
# TestFinalSelect
# ====================================================================
class TestFinalSelect:
    """最終選択（マルチ解像度ランク集約）の動作を検証するテストクラス。"""

    def test_returns_single_tensor(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """最終選択がリストではなく単一のテンソルを返すことを検証する。"""
        # 全ステージを実行して候補を準備
        stage1_candidates = small_decoder._stage1_prune(mock_model, prefix)
        stage2_candidates = small_decoder._stage2_prune(
            mock_model, stage1_candidates
        )
        stage3_candidates = small_decoder._stage3_extend(
            mock_model, stage2_candidates
        )

        result = small_decoder._final_select(stage3_candidates)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1, (
            f"最終選択の結果は1Dテンソルであるべき: got dim={result.dim()}"
        )

    def test_selected_from_candidates(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
        prefix: torch.Tensor,
    ):
        """返されたテンソルが入力候補のいずれかと一致することを検証する。"""
        stage1_candidates = small_decoder._stage1_prune(mock_model, prefix)
        stage2_candidates = small_decoder._stage2_prune(
            mock_model, stage1_candidates
        )
        stage3_candidates = small_decoder._stage3_extend(
            mock_model, stage2_candidates
        )

        result = small_decoder._final_select(stage3_candidates)

        # _final_select は末尾L3トークンを返す
        # 候補の末尾L3トークンのいずれかと一致すること
        L3 = small_decoder.L3
        match_found = any(
            torch.equal(result, cand[-L3:]) for cand in stage3_candidates
        )
        assert match_found, (
            "最終選択の結果が入力候補の末尾L3トークンのいずれとも一致しない"
        )


# ====================================================================
# TestEndToEnd
# ====================================================================
class TestEndToEnd:
    """エンドツーエンドのgenerate()呼び出しを検証するテストクラス。"""

    def test_generate_runs_without_error(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
    ):
        """generate()がエラーなく完了することを検証する。"""
        text_tokens = torch.randint(0, VOCAB_SIZE, (10,))
        speaker_tokens = torch.randint(0, VOCAB_SIZE, (5,))

        result = small_decoder.generate(
            model=mock_model,
            text_tokens=text_tokens,
            speaker_tokens=speaker_tokens,
        )
        assert isinstance(result, torch.Tensor)

    def test_output_is_longer_than_input(
        self,
        small_decoder: HierarchicalDecoder,
        mock_model: MockBackboneModel,
    ):
        """生成されたトークン列が入力より長いことを検証する。"""
        text_tokens = torch.randint(0, VOCAB_SIZE, (10,))
        speaker_tokens = torch.randint(0, VOCAB_SIZE, (5,))
        input_length = len(text_tokens) + len(speaker_tokens)

        result = small_decoder.generate(
            model=mock_model,
            text_tokens=text_tokens,
            speaker_tokens=speaker_tokens,
        )
        assert len(result) > input_length, (
            f"出力が入力より長くない: output len={len(result)}, "
            f"input len={input_length}"
        )

    def test_generate_with_small_max_length(
        self,
        eas_sampler: EntropyAwareSampler,
        detectors: MultiResolutionDetector,
        mock_model: MockBackboneModel,
    ):
        """max_lengthが尊重されることを検証する。"""
        max_length = 30
        decoder = HierarchicalDecoder(
            eas_sampler=eas_sampler,
            detectors=detectors,
            warmup=SMALL_WARMUP,
            stage_lengths=SMALL_STAGE_LENGTHS,
            beam_sizes=SMALL_BEAM_SIZES,
            max_length=max_length,
        )
        text_tokens = torch.randint(0, VOCAB_SIZE, (5,))
        speaker_tokens = torch.randint(0, VOCAB_SIZE, (3,))

        result = decoder.generate(
            model=mock_model,
            text_tokens=text_tokens,
            speaker_tokens=speaker_tokens,
        )
        # generate はL3トークン単位で生成するため、max_length + L3 までは許容
        L3 = SMALL_STAGE_LENGTHS[2]
        assert len(result) <= max_length + L3, (
            f"max_lengthを大幅に超過: output len={len(result)}, "
            f"max_length={max_length}, L3={L3}"
        )
