"""Entropy-Aware Sampling (EAS) の実装 — Algorithm 1。

VALL-E 2 の Repetition-Aware Sampling (RAS) を改良したデコーディング戦略。
メモリバッファを使用して候補トークンのランク位置と時間的経過を記録し、
逆ランク重み付けと指数的時間減衰によるペナルティを適用する。

ハイパーパラメータ:
    - top_k: 50
    - top_p (v): 0.8
    - temperature: 1.0
    - cluster_size (k_e): 3
    - memory_window (W): 15
    - alpha: 0.2 (ペナルティスケール)
    - beta: 0.7 (時間減衰率)
    - gamma: 0.8 (クリッピング上限)
"""

from dataclasses import dataclass
import math

import torch

from mspoof_tts.sampling.utils import (
    apply_temperature,
    top_k_filter,
    nucleus_sample,
    compute_entropy,
    rank_tokens,
)


@dataclass
class MemoryEntry:
    """メモリバッファの1エントリ。

    Attributes:
        token_id: トークンの識別子。
        rank: 確率分布におけるランク（0始まり、高確率ほど低ランク）。
        age: 記録からの経過ステップ数。
    """

    token_id: int
    rank: int
    age: int = 0


class MemoryBuffer:
    """EASのメモリバッファ。

    生成済みトークンのランク情報と経過時間を保持し、
    ウィンドウサイズを超えた古いエントリを自動的に削除する。

    Args:
        window_size: メモリの保持ステップ数 (W)。
    """

    def __init__(self, window_size: int = 15) -> None:
        self.window_size = window_size
        self.entries: list[MemoryEntry] = []

    def add(self, entries: list[MemoryEntry]) -> None:
        """新しいエントリをメモリに追加し、agingを実行する。

        Args:
            entries: 追加するメモリエントリのリスト。
        """
        # 既存エントリのageをインクリメント
        for entry in self.entries:
            entry.age += 1

        # 新しいエントリを追加
        self.entries.extend(entries)

        # ウィンドウサイズを超えた古いエントリを削除
        self.entries = [e for e in self.entries if e.age <= self.window_size]

    def get_penalty(
        self, vocab_size: int, alpha: float, beta: float, gamma: float
    ) -> torch.Tensor:
        """メモリに基づくペナルティベクトルを計算する。

        penalty(j) = min(gamma, sum_{(i,r,a) in M, i=j} alpha * (1/(1+r)) * beta^a)

        Args:
            vocab_size: 語彙サイズ。
            alpha: ペナルティスケール。
            beta: 時間減衰率。
            gamma: クリッピング上限。

        Returns:
            ペナルティベクトル (vocab_size,)。
        """
        penalty = torch.zeros(vocab_size)

        if not self.entries:
            return penalty

        # ベクトル化: エントリをテンソルに変換し scatter_add で集約
        token_ids = torch.tensor(
            [e.token_id for e in self.entries], dtype=torch.long
        )
        ranks = torch.tensor(
            [e.rank for e in self.entries], dtype=torch.float
        )
        ages = torch.tensor(
            [e.age for e in self.entries], dtype=torch.float
        )

        # 各エントリのペナルティ寄与: alpha * (1/(1+r)) * beta^a
        contributions = alpha * (1.0 / (1.0 + ranks)) * (beta ** ages)

        # token_id ごとに寄与を集約
        penalty.scatter_add_(0, token_ids, contributions)

        # クリッピング
        penalty = torch.clamp(penalty, max=gamma)

        return penalty


class EntropyAwareSampler:
    """Entropy-Aware Samplingによるトークン生成器。

    ARモデルの確率分布にメモリベースのペナルティを適用し、
    nucleus samplingでトークンを生成する。

    Args:
        top_k: top-kフィルタリングのk値。
        top_p: nucleus samplingの累積確率閾値。
        temperature: サンプリング温度。
        cluster_size: メモリに記録するトップトークン数 (k_e)。
        memory_window: メモリの保持ステップ数 (W)。
        alpha: ペナルティスケール。
        beta: 時間減衰率。
        gamma: クリッピング上限。
    """

    def __init__(
        self,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 1.0,
        cluster_size: int = 3,
        memory_window: int = 15,
        alpha: float = 0.2,
        beta: float = 0.7,
        gamma: float = 0.8,
    ) -> None:
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.cluster_size = cluster_size
        self.memory_window = memory_window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.memory = MemoryBuffer(memory_window)

    def reset(self) -> None:
        """メモリバッファをリセットし、新しい生成を開始する。"""
        self.memory = MemoryBuffer(self.memory_window)

    def sample_step(self, logits: torch.Tensor) -> torch.Tensor:
        """1ステップのEASサンプリングを実行する。

        手順:
            1. logitsからsoftmaxで確率分布を取得
            2. メモリベースのペナルティを計算・適用
            3. nucleus samplingでトークンを生成
            4. トップk_eトークンをクラスタとしてメモリに追加

        Args:
            logits: ARモデルからの生のlogits (vocab_size,) または (batch, vocab_size)。

        Returns:
            サンプリングされたトークンID。
        """
        # バッチ次元の正規化: 1Dまたはbatch=1に統一
        squeezed = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeezed = True
        # batch=1のみサポート
        logits = logits[0]  # (vocab_size,)

        vocab_size = logits.size(-1)

        # ステップ1: 温度スケーリング + softmaxで確率分布を取得
        probs = torch.softmax(logits / self.temperature, dim=-1)

        # ステップ2: エントロピーの計算
        H = compute_entropy(probs)

        # ステップ3: ランクの計算
        ranks = rank_tokens(logits)

        # ステップ4: メモリベースペナルティの計算
        penalty = self.memory.get_penalty(
            vocab_size, self.alpha, self.beta, self.gamma
        )
        penalty = penalty.to(logits.device)

        # ステップ5: エントロピーに基づく条件付きペナルティ適用
        # エントロピーが高い場合（H > log(k_e)）のみペナルティを適用
        entropy_threshold = math.log(self.cluster_size)
        if H.item() > entropy_threshold:
            adjusted_probs = probs - penalty
        else:
            adjusted_probs = probs.clone()

        # ステップ6: 再正規化
        adjusted_probs = torch.clamp(adjusted_probs, min=0.0)
        prob_sum = adjusted_probs.sum()
        if prob_sum > 0:
            adjusted_probs = adjusted_probs / prob_sum
        else:
            # 全てゼロになった場合は元の確率分布にフォールバック
            adjusted_probs = probs

        # ステップ7: top-kフィルタ + nucleus sampling
        # 調整済み確率をlogitsに変換
        adjusted_logits = torch.log(adjusted_probs + 1e-12)
        # top-kフィルタリング
        filtered_logits = top_k_filter(adjusted_logits, self.top_k)
        # nucleus sampling（温度は既に適用済みなので1.0を使用）
        token_id = nucleus_sample(filtered_logits, self.top_p, temperature=1.0)

        # ステップ8: メモリ更新 — トップk_eトークンをクラスタとしてメモリに追加
        # 調整済み分布からトップk_eトークンを取得
        k_e = min(self.cluster_size, vocab_size)
        top_k_values, top_k_indices = torch.topk(adjusted_probs, k_e)
        new_entries = [
            MemoryEntry(
                token_id=top_k_indices[r].item(),
                rank=r,
                age=0,
            )
            for r in range(k_e)
        ]
        self.memory.add(new_entries)

        # ステップ9: サンプリングされたトークンIDを返す
        if squeezed:
            return token_id
        return token_id.unsqueeze(0)

    def generate(
        self,
        model: object,
        prefix: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """EASを使用して複数トークンを連続生成する。

        Args:
            model: 次トークン予測を行うARモデル。
            prefix: プレフィックストークン列 (seq_len,)。
            num_tokens: 生成するトークン数。

        Returns:
            生成されたトークン列 (num_tokens,)。
        """
        self.reset()
        generated = []
        current_input = prefix.clone()

        for _ in range(num_tokens):
            # モデルから次トークンのlogitsを取得
            logits = model(current_input)  # type: ignore[operator]
            token = self.sample_step(logits)
            generated.append(token)
            # トークンを入力列に追加
            current_input = torch.cat(
                [current_input, token.reshape(1)], dim=-1
            )

        return torch.stack(generated)
