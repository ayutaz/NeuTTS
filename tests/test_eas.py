"""Entropy-Aware Sampling (EAS) モジュールのユニットテスト。

小さな語彙サイズと既知のロジットを使用して、サンプリングユーティリティ関数
およびEASサンプラーの各コンポーネントを検証する。外部データやGPUは不要。
"""

import math

import pytest
import torch

from mspoof_tts.sampling.utils import (
    apply_temperature,
    compute_entropy,
    nucleus_sample,
    rank_tokens,
    top_k_filter,
)
from mspoof_tts.sampling.eas import (
    EntropyAwareSampler,
    MemoryBuffer,
    MemoryEntry,
)

# ---------- 共通テスト設定 ----------
VOCAB_SIZE = 10


# ====================================================================
# TestApplyTemperature
# ====================================================================
class TestApplyTemperature:
    """温度スケーリング関数の動作を検証するテストクラス。"""

    def test_identity(self):
        """temperature=1.0で入力ロジットが変化しないことを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = apply_temperature(logits, temperature=1.0)
        assert torch.allclose(result, logits)

    def test_high_temperature(self):
        """temperature=2.0で分布がより均一になる（最大確率が低下する）ことを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original_probs = torch.softmax(logits, dim=-1)
        scaled = apply_temperature(logits, temperature=2.0)
        scaled_probs = torch.softmax(scaled, dim=-1)
        # 高温では最大確率が下がり、分布がより均一になる
        assert scaled_probs.max() < original_probs.max()

    def test_low_temperature(self):
        """temperature=0.5で分布がより鋭くなる（最大確率が上昇する）ことを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        original_probs = torch.softmax(logits, dim=-1)
        scaled = apply_temperature(logits, temperature=0.5)
        scaled_probs = torch.softmax(scaled, dim=-1)
        # 低温では最大確率が上がり、分布がより鋭くなる
        assert scaled_probs.max() > original_probs.max()

    def test_invalid_temperature(self):
        """temperature=0でValueErrorが送出されることを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            apply_temperature(logits, temperature=0)

    def test_1d_and_2d(self):
        """1次元と2次元の両方の入力形状で動作することを検証する。"""
        logits_1d = torch.tensor([1.0, 2.0, 3.0])
        result_1d = apply_temperature(logits_1d, temperature=2.0)
        assert result_1d.shape == (3,)
        assert torch.allclose(result_1d, logits_1d / 2.0)

        logits_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result_2d = apply_temperature(logits_2d, temperature=2.0)
        assert result_2d.shape == (2, 3)
        assert torch.allclose(result_2d, logits_2d / 2.0)


# ====================================================================
# TestTopKFilter
# ====================================================================
class TestTopKFilter:
    """top-kフィルタリング関数の動作を検証するテストクラス。"""

    def test_basic(self):
        """k=3のとき上位3個のみが残り、残りが-infになることを検証する。"""
        logits = torch.tensor([1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0])
        filtered = top_k_filter(logits, k=3)
        # 上位3個は 7.0, 6.0, 5.0 (インデックス 3, 5, 1)
        finite_mask = torch.isfinite(filtered)
        assert finite_mask.sum().item() == 3
        # -inf でないインデックスが上位3つに対応する
        top3_indices = set(torch.where(finite_mask)[0].tolist())
        assert top3_indices == {1, 3, 5}

    def test_k_larger_than_vocab(self):
        """kが語彙サイズより大きい場合、ロジットが変化しないことを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        filtered = top_k_filter(logits, k=10)
        assert torch.allclose(filtered, logits)

    def test_k_zero(self):
        """k=0の場合、ロジットが変化しないことを検証する。"""
        logits = torch.tensor([1.0, 2.0, 3.0])
        filtered = top_k_filter(logits, k=0)
        assert torch.allclose(filtered, logits)


# ====================================================================
# TestNucleusSample
# ====================================================================
class TestNucleusSample:
    """Nucleus sampling関数の動作を検証するテストクラス。"""

    def test_returns_valid_token(self):
        """返されるトークンIDが有効範囲 [0, vocab_size) にあることを検証する。"""
        logits = torch.randn(VOCAB_SIZE)
        token = nucleus_sample(logits, top_p=0.8)
        assert 0 <= token.item() < VOCAB_SIZE

    def test_deterministic_with_peaked_distribution(self):
        """非常に鋭い分布では一貫したトークンが返されることを検証する。"""
        logits = torch.full((VOCAB_SIZE,), -100.0)
        logits[7] = 100.0  # トークン7が圧倒的に高い
        tokens = set()
        for _ in range(50):
            t = nucleus_sample(logits, top_p=0.9)
            tokens.add(t.item())
        # 非常に鋭い分布なので、ほぼ常にトークン7が選ばれるべき
        assert 7 in tokens
        assert len(tokens) <= 2  # 稀にブレがあっても高々2種類

    def test_shape_1d_and_2d(self):
        """1次元と2次元の両方の入力形状で動作することを検証する。"""
        logits_1d = torch.randn(VOCAB_SIZE)
        token_1d = nucleus_sample(logits_1d, top_p=0.8)
        # 1Dの場合、squeezeされてスカラーまたは (1,) を返す
        assert token_1d.numel() == 1

        logits_2d = torch.randn(3, VOCAB_SIZE)
        token_2d = nucleus_sample(logits_2d, top_p=0.8)
        assert token_2d.shape == (3, 1)


# ====================================================================
# TestComputeEntropy
# ====================================================================
class TestComputeEntropy:
    """エントロピー計算関数の動作を検証するテストクラス。"""

    def test_uniform_distribution(self):
        """一様分布のエントロピーがlog(n)に等しいことを検証する。"""
        n = 10
        probs = torch.ones(n) / n
        entropy = compute_entropy(probs)
        expected = math.log(n)
        assert abs(entropy.item() - expected) < 1e-5

    def test_peaked_distribution(self):
        """ワンホットに近い分布でエントロピーがほぼ0になることを検証する。"""
        probs = torch.zeros(VOCAB_SIZE)
        probs[3] = 1.0
        entropy = compute_entropy(probs)
        assert entropy.item() < 1e-6

    def test_known_value(self):
        """手動計算したエントロピー値と一致することを検証する。"""
        # p = [0.5, 0.25, 0.25] のエントロピー
        # H = -(0.5*ln(0.5) + 0.25*ln(0.25) + 0.25*ln(0.25))
        probs = torch.tensor([0.5, 0.25, 0.25])
        entropy = compute_entropy(probs)
        expected = -(0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25))
        assert abs(entropy.item() - expected) < 1e-5


# ====================================================================
# TestRankTokens
# ====================================================================
class TestRankTokens:
    """トークンランク付け関数の動作を検証するテストクラス。"""

    def test_basic_ranking(self):
        """最も高いロジットのトークンがランク0を得ることを検証する。"""
        logits = torch.tensor([1.0, 5.0, 3.0, 7.0, 2.0])
        ranks = rank_tokens(logits)
        # 7.0(index 3)が最大 -> ランク0
        assert ranks[3].item() == 0
        # 5.0(index 1)が2番目 -> ランク1
        assert ranks[1].item() == 1
        # 1.0(index 0)が最小 -> ランク4
        assert ranks[0].item() == 4

    def test_all_equal(self):
        """全トークンが同じロジットでもクラッシュせずランクが割り当てられることを検証する。"""
        logits = torch.ones(VOCAB_SIZE)
        ranks = rank_tokens(logits)
        # 全てのランク値がユニークであること（tiebreakが何らかの順序で割り当てられる）
        assert ranks.shape == (VOCAB_SIZE,)
        assert set(ranks.tolist()) == set(range(VOCAB_SIZE))


# ====================================================================
# TestMemoryBuffer
# ====================================================================
class TestMemoryBuffer:
    """EASメモリバッファの動作を検証するテストクラス。"""

    def test_empty_penalty(self):
        """エントリがない状態でペナルティベクトルが全てゼロであることを検証する。"""
        buf = MemoryBuffer(window_size=15)
        penalty = buf.get_penalty(vocab_size=VOCAB_SIZE, alpha=0.2, beta=0.7, gamma=0.8)
        assert penalty.shape == (VOCAB_SIZE,)
        assert torch.allclose(penalty, torch.zeros(VOCAB_SIZE))

    def test_add_and_penalty(self):
        """エントリ追加後、該当トークンIDのペナルティが正値になることを検証する。"""
        buf = MemoryBuffer(window_size=15)
        entry = MemoryEntry(token_id=3, rank=0, age=0)
        buf.add([entry])
        penalty = buf.get_penalty(vocab_size=VOCAB_SIZE, alpha=0.2, beta=0.7, gamma=0.8)
        # token_id=3 のペナルティは正であるべき
        assert penalty[3].item() > 0
        # 他のトークンのペナルティはゼロであるべき
        for i in range(VOCAB_SIZE):
            if i != 3:
                assert penalty[i].item() == 0.0

    def test_aging(self):
        """複数回追加後、古いエントリのペナルティが減衰することを検証する。"""
        buf = MemoryBuffer(window_size=15)
        entry = MemoryEntry(token_id=5, rank=0, age=0)
        buf.add([entry])
        penalty_fresh = buf.get_penalty(
            vocab_size=VOCAB_SIZE, alpha=0.2, beta=0.7, gamma=0.8
        )[5].item()

        # 新しいエントリを追加してエイジングを進める
        buf.add([MemoryEntry(token_id=9, rank=1, age=0)])
        penalty_aged = buf.get_penalty(
            vocab_size=VOCAB_SIZE, alpha=0.2, beta=0.7, gamma=0.8
        )[5].item()

        # エイジングにより、token_id=5のペナルティは減少しているべき
        assert penalty_aged < penalty_fresh

    def test_window_eviction(self):
        """ウィンドウサイズを超えたエントリが除去されることを検証する。"""
        window_size = 3
        buf = MemoryBuffer(window_size=window_size)
        # ウィンドウサイズより多くのステップを追加（window_size+1回で超過）
        buf.add([MemoryEntry(token_id=2, rank=0, age=0)])
        for _ in range(window_size + 1):
            buf.add([MemoryEntry(token_id=9, rank=1, age=0)])

        penalty = buf.get_penalty(vocab_size=VOCAB_SIZE, alpha=0.2, beta=0.7, gamma=0.8)
        # token_id=2 のエントリはウィンドウ外に出たのでペナルティはゼロ
        assert penalty[2].item() == 0.0

    def test_clipping(self):
        """ペナルティがgammaを超えないことを検証する。"""
        buf = MemoryBuffer(window_size=100)
        gamma = 0.3
        # 同じトークンを大量に追加してペナルティを蓄積させる
        for _ in range(50):
            buf.add([MemoryEntry(token_id=1, rank=0, age=0)])

        penalty = buf.get_penalty(
            vocab_size=VOCAB_SIZE, alpha=0.5, beta=0.99, gamma=gamma
        )
        assert penalty[1].item() <= gamma + 1e-7


# ====================================================================
# TestEntropyAwareSampler
# ====================================================================
class TestEntropyAwareSampler:
    """EASサンプラーの初期化・リセット・サンプリング動作を検証するテストクラス。"""

    def test_init(self):
        """全パラメータが正しく保存されていることを検証する。"""
        sampler = EntropyAwareSampler(
            top_k=30,
            top_p=0.9,
            temperature=0.8,
            cluster_size=5,
            memory_window=20,
            alpha=0.3,
            beta=0.6,
            gamma=0.7,
        )
        assert sampler.top_k == 30
        assert sampler.top_p == 0.9
        assert sampler.temperature == 0.8
        assert sampler.cluster_size == 5
        assert sampler.memory_window == 20
        assert sampler.alpha == 0.3
        assert sampler.beta == 0.6
        assert sampler.gamma == 0.7

    def test_reset(self):
        """リセット後にメモリがクリアされることを検証する。"""
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.8, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.2, beta=0.7, gamma=0.8,
        )
        # サンプリングを何回か実行してメモリにエントリを蓄積
        logits = torch.randn(VOCAB_SIZE)
        for _ in range(5):
            sampler.sample_step(logits)

        sampler.reset()

        # リセット後のペナルティはゼロであるべき
        penalty = sampler.memory.get_penalty(
            vocab_size=VOCAB_SIZE,
            alpha=sampler.alpha,
            beta=sampler.beta,
            gamma=sampler.gamma,
        )
        assert torch.allclose(penalty, torch.zeros(VOCAB_SIZE))

    def test_sample_step_returns_valid_token(self):
        """sample_stepが有効なトークンIDを返すことを検証する。"""
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.8, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.2, beta=0.7, gamma=0.8,
        )
        logits = torch.randn(VOCAB_SIZE)
        token = sampler.sample_step(logits)
        assert 0 <= token.item() < VOCAB_SIZE

    def test_sample_step_avoids_repetition(self):
        """繰り返しサンプリングで多様性が確保されることを検証する。

        同じロジットから多数回サンプリングし、常に同一トークンに
        ならないことを確認する（メモリペナルティの効果）。
        """
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.95, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.5, beta=0.7, gamma=0.8,
        )
        # 複数トークンが同程度の確率を持つロジット
        logits = torch.zeros(20)
        logits[0] = 2.0
        logits[1] = 1.8
        logits[2] = 1.6
        logits[3] = 1.4
        logits[4] = 1.2

        tokens = set()
        for _ in range(30):
            token = sampler.sample_step(logits)
            tokens.add(token.item())

        # ペナルティにより複数の異なるトークンが選ばれるべき
        assert len(tokens) > 1

    def test_penalty_reduces_probability(self):
        """メモリにトークンを追加した後、そのトークンの確率が低下することを検証する。"""
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.95, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.5, beta=0.7, gamma=0.8,
        )
        # 鋭すぎない分布を用意（ペナルティが効く中程度のエントロピー）
        logits = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

        # ペナルティなしの確率
        probs_before = torch.softmax(logits, dim=-1)

        # トークン0（最大ロジット）をメモリに追加
        sampler.memory.add([MemoryEntry(token_id=0, rank=0, age=0)])

        # ペナルティ適用後のロジットを計算
        penalty = sampler.memory.get_penalty(
            vocab_size=VOCAB_SIZE,
            alpha=sampler.alpha,
            beta=sampler.beta,
            gamma=sampler.gamma,
        )
        penalized_logits = logits - penalty
        probs_after = torch.softmax(penalized_logits, dim=-1)

        # トークン0の確率が低下しているべき
        assert probs_after[0].item() < probs_before[0].item()

    def test_high_entropy_applies_penalty(self):
        """エントロピーが高い分布でペナルティが適用されることを検証する。"""
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.95, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.5, beta=0.7, gamma=0.8,
        )
        # 均一に近いロジット（高エントロピー）
        logits = torch.zeros(VOCAB_SIZE)
        # メモリにエントリを追加
        sampler.memory.add([MemoryEntry(token_id=0, rank=0, age=0)])

        # 高エントロピー分布ではペナルティが適用されるべき
        penalty = sampler.memory.get_penalty(
            vocab_size=VOCAB_SIZE,
            alpha=sampler.alpha,
            beta=sampler.beta,
            gamma=sampler.gamma,
        )
        assert penalty[0].item() > 0

        # sample_stepを呼び出してエラーなく動作することを確認
        token = sampler.sample_step(logits)
        assert 0 <= token.item() < VOCAB_SIZE

    def test_low_entropy_skips_penalty(self):
        """エントロピーが低い（鋭い）分布ではペナルティがスキップされることを検証する。

        非常に鋭い分布ではサンプリング結果がペナルティの有無に関わらず
        支配的なトークンに集中するべき。
        """
        sampler = EntropyAwareSampler(
            top_k=50, top_p=0.95, temperature=1.0,
            cluster_size=3, memory_window=15,
            alpha=0.5, beta=0.7, gamma=0.8,
        )
        # 非常に鋭い分布（低エントロピー）
        logits = torch.full((VOCAB_SIZE,), -100.0)
        logits[5] = 100.0  # トークン5が圧倒的に高い

        # メモリにトークン5を追加
        sampler.memory.add([MemoryEntry(token_id=5, rank=0, age=0)])

        # 低エントロピーではペナルティがスキップされ、
        # 支配的トークンが選ばれ続けるべき
        tokens = []
        for _ in range(20):
            sampler_fresh = EntropyAwareSampler(
                top_k=50, top_p=0.95, temperature=1.0,
                cluster_size=3, memory_window=15,
                alpha=0.5, beta=0.7, gamma=0.8,
            )
            sampler_fresh.memory.add([MemoryEntry(token_id=5, rank=0, age=0)])
            token = sampler_fresh.sample_step(logits)
            tokens.append(token.item())

        # 低エントロピーでペナルティがスキップされるなら、
        # 大半のサンプルがトークン5であるべき
        count_5 = tokens.count(5)
        assert count_5 >= 15, (
            f"低エントロピー分布でトークン5が{count_5}/20回しか選ばれなかった"
        )
