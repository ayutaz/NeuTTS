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
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}"
        )
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """top-kフィルタリングを適用する。

    上位k個以外のロジットを -inf に設定する。

    Args:
        logits: ロジット (batch, vocab_size) または (vocab_size,)。
        k: 保持するトークン数。

    Returns:
        フィルタリング済みロジット。
    """
    if k <= 0:
        return logits
    vocab_size = logits.size(-1)
    if k >= vocab_size:
        return logits
    top_k_val = torch.topk(logits, k, dim=-1).values[..., -1:]
    logits = logits.masked_fill(logits < top_k_val, float('-inf'))
    return logits


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
    squeezed = logits.dim() == 1
    if squeezed:
        logits = logits.unsqueeze(0)

    logits = apply_temperature(logits, temperature)

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Mask tokens where cumulative probability exceeds top_p,
    # but always keep at least the top token.
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    # Sample from the filtered distribution.
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(probs, num_samples=1)

    # Map back to original token indices.
    token_ids = sorted_indices.gather(dim=-1, index=sampled_sorted)

    if squeezed:
        token_ids = token_ids.squeeze(0)
    return token_ids


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """確率分布のエントロピーを計算する。

    H(p) = -sum(p * log(p))

    Args:
        probs: 確率分布 (batch, vocab_size) または (vocab_size,)。

    Returns:
        エントロピー値。
    """
    clamped = torch.clamp(probs, min=1e-12)
    return -(probs * torch.log(clamped)).sum(dim=-1)


def rank_tokens(logits: torch.Tensor) -> torch.Tensor:
    """ロジットに基づいてトークンのランクを算出する。

    最も高いロジットのトークンがランク0となる。

    Args:
        logits: ロジット (batch, vocab_size) または (vocab_size,)。

    Returns:
        各トークンのランク。形状は入力と同じ。
    """
    return torch.argsort(
        torch.argsort(logits, dim=-1, descending=True), dim=-1
    )
