"""セグメント構築モジュール。

トークン列からスプーフ検出器の入力セグメントを切り出す。
以下の2つの方式をサポートする:

1. Contiguous cropping: 連続するLトークンを切り出す
   - M_10: L=10
   - M_25: L=25
   - M_50: L=50

2. Skip sampling: スパンからスキップ率rで間引いてサンプリングする
   - M_50_25: 50トークンスパンからr=2でサンプリング (25トークン選択)
   - M_50_10: 50トークンスパンからr=5でサンプリング (10トークン選択)
"""

from enum import Enum
from typing import Optional

import torch


class SegmentMode(Enum):
    """セグメント構築方式の列挙型。"""

    CONTIGUOUS_10 = "contiguous_10"
    CONTIGUOUS_25 = "contiguous_25"
    CONTIGUOUS_50 = "contiguous_50"
    SKIP_50_25 = "skip_50_25"
    SKIP_50_10 = "skip_50_10"


class SegmentExtractor:
    """トークン列からセグメントを抽出するクラス。

    指定されたモードに応じて、contiguous croppingまたはskip samplingで
    セグメントを切り出す。

    Args:
        mode: セグメント構築方式。
        random_offset: Trueの場合、切り出し開始位置をランダムに選択する。
    """

    def __init__(self, mode: SegmentMode, random_offset: bool = True) -> None:
        raise NotImplementedError("SegmentExtractor の実装が必要")

    def extract(self, token_ids: torch.Tensor) -> torch.Tensor:
        """トークン列からセグメントを抽出する。

        Args:
            token_ids: 入力トークンID列 (seq_len,)。

        Returns:
            抽出されたセグメント (segment_len,)。
            contiguous_10 -> (10,), contiguous_25 -> (25,),
            contiguous_50 -> (50,), skip_50_25 -> (25,),
            skip_50_10 -> (10,)。
        """
        raise NotImplementedError

    @staticmethod
    def contiguous_crop(
        token_ids: torch.Tensor, length: int, offset: Optional[int] = None
    ) -> torch.Tensor:
        """連続するトークンを切り出す (contiguous cropping)。

        Args:
            token_ids: 入力トークンID列 (seq_len,)。
            length: 切り出すトークン数。
            offset: 切り出し開始位置。Noneの場合ランダム。

        Returns:
            切り出されたセグメント (length,)。
        """
        raise NotImplementedError

    @staticmethod
    def skip_sample(
        token_ids: torch.Tensor,
        span: int,
        skip_rate: int,
        offset: Optional[int] = None,
    ) -> torch.Tensor:
        """スキップサンプリングでトークンを抽出する。

        spanトークンのスパンからskip_rateごとにトークンを間引いて選択する。

        Args:
            token_ids: 入力トークンID列 (seq_len,)。
            span: スパンのトークン数 (例: 50)。
            skip_rate: スキップ率 (例: 2 or 5)。
            offset: スパン開始位置。Noneの場合ランダム。

        Returns:
            スキップサンプリングされたセグメント (span // skip_rate,)。
        """
        raise NotImplementedError
