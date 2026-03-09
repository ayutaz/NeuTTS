"""スプーフ検出器訓練用データセット。

real（実音声由来）とfake（合成音声由来）の離散トークンセグメントペアを
提供するPyTorchデータセット。各セグメントには二値ラベル（real=1, fake=0）が
付与される。
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from mspoof_tts.data.segment import SegmentExtractor


class SpoofDetectionDataset(Dataset):
    """スプーフ検出器訓練用データセット。

    メタデータファイルからreal/fakeトークン列のパスとラベルを読み込み、
    指定された解像度設定でセグメントを切り出して返す。

    Args:
        metadata_path: メタデータファイルのパス。
        segment_extractor: セグメント抽出器。
        split: データ分割 ("train", "val", "test")。
        max_samples: 最大サンプル数（デバッグ用）。
    """

    def __init__(
        self,
        metadata_path: Path,
        segment_extractor: SegmentExtractor,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        raise NotImplementedError("SpoofDetectionDataset の実装が必要")

    def __len__(self) -> int:
        """データセットのサンプル数を返す。"""
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """指定インデックスのサンプルを返す。

        Returns:
            以下のキーを持つ辞書:
                - "token_ids": セグメントのトークンID列 (segment_len,)
                - "label": real=1, fake=0 のラベル (スカラ)
                - "utterance_id": 発話ID (文字列)
        """
        raise NotImplementedError
