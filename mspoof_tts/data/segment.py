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

    # モードごとのパラメータ: (segment_length, span, skip_rate)
    # contiguous モードでは span=None, skip_rate=None
    _MODE_PARAMS: dict[SegmentMode, tuple[int, Optional[int], Optional[int]]] = {
        SegmentMode.CONTIGUOUS_10: (10, None, None),
        SegmentMode.CONTIGUOUS_25: (25, None, None),
        SegmentMode.CONTIGUOUS_50: (50, None, None),
        SegmentMode.SKIP_50_25: (25, 50, 2),
        SegmentMode.SKIP_50_10: (10, 50, 5),
    }

    def __init__(self, mode: SegmentMode, random_offset: bool = True) -> None:
        self.mode = mode
        self.random_offset = random_offset

        params = self._MODE_PARAMS[mode]
        self.segment_length: int = params[0]
        self.span: Optional[int] = params[1]
        self.skip_rate: Optional[int] = params[2]

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
        offset = None if self.random_offset else 0

        if self.span is not None:
            # skip sampling モード
            required = self.span
            if token_ids.shape[0] < required:
                raise ValueError(
                    f"トークン列が短すぎます: {token_ids.shape[0]} < "
                    f"必要なスパン長 {required} (モード: {self.mode.value})"
                )
            return self.skip_sample(
                token_ids, self.span, self.skip_rate, offset=offset  # type: ignore[arg-type]
            )
        else:
            # contiguous cropping モード
            required = self.segment_length
            if token_ids.shape[0] < required:
                raise ValueError(
                    f"トークン列が短すぎます: {token_ids.shape[0]} < "
                    f"必要なセグメント長 {required} (モード: {self.mode.value})"
                )
            return self.contiguous_crop(token_ids, self.segment_length, offset=offset)

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
        seq_len = token_ids.shape[0]
        if offset is None:
            max_offset = seq_len - length
            offset = torch.randint(0, max_offset + 1, (1,)).item()
        return token_ids[offset : offset + length]

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
        seq_len = token_ids.shape[0]
        if offset is None:
            max_offset = seq_len - span
            offset = torch.randint(0, max_offset + 1, (1,)).item()
        span_tokens = token_ids[offset : offset + span]
        return span_tokens[::skip_rate]

    def get_output_length(self) -> int:
        """現在のモードにおける出力セグメントのトークン数を返す。

        Returns:
            出力セグメントのトークン数。
        """
        return self.segment_length
