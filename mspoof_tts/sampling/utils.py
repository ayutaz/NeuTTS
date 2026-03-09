"""サンプリングユーティリティ関数。

Nucleus sampling (top-p)、top-kフィルタリング、温度スケーリングなど、
サンプリングに共通する基本操作を提供する。
"""

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """ロジットに温度スケーリングを適用する。

    Args:
        logits: 生のロジット (batch, vocab_size) または (vocab_size,)。
        temperature: サンプリング温度。1.0で変化なし。

    Returns:
        温度スケーリング済みロジット。
    """
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """top-kフィルタリングを適用する。

    上位k個以外のロジットを -inf に設定する。

    Args:
        logits: ロジット (batch, vocab_size) または (vocab_size,)。
        k: 保持するトークン数。

    Returns:
        フィルタリング済みロジット。
    """
    raise NotImplementedError


def nucleus_sample(
    logits: torch.Tensor, top_p: float, temperature: float = 1.0
) -> torch.Tensor:
    """Nucleus sampling (top-p sampling) を実行する。

    累積確率がtop_pを超えるまでのトークンのみを候補として
    サンプリングする。

    Args:
        logits: ロジット (batch, vocab_size) または (vocab_size,)。
        top_p: 累積確率の閾値。
        temperature: サンプリング温度。

    Returns:
        サンプリングされたトークンID。
    """
    raise NotImplementedError


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """確率分布のエントロピーを計算する。

    H(p) = -sum(p * log(p))

    Args:
        probs: 確率分布 (batch, vocab_size) または (vocab_size,)。

    Returns:
        エントロピー値。
    """
    raise NotImplementedError


def rank_tokens(logits: torch.Tensor) -> torch.Tensor:
    """ロジットに基づいてトークンのランクを算出する。

    最も高いロジットのトークンがランク0となる。

    Args:
        logits: ロジット (batch, vocab_size) または (vocab_size,)。

    Returns:
        各トークンのランク。形状は入力と同じ。
    """
    raise NotImplementedError
