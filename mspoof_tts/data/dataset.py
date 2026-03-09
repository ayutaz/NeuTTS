"""スプーフ検出器訓練用データセット。

real（実音声由来）とfake（合成音声由来）の離散トークンセグメントペアを
提供するPyTorchデータセット。各セグメントには二値ラベル（real=1, fake=0）が
付与される。
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

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
        self.metadata_path = Path(metadata_path)
        self.base_dir = self.metadata_path.parent
        self.segment_extractor = segment_extractor
        self.split = split

        # メタデータファイルを読み込み、指定されたsplitでフィルタリングする
        self.samples: list[dict] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("split") == split:
                    self.samples.append(entry)

        # 最大サンプル数でトランケートする（デバッグ用）
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        # ラベルごとのインデックスマップを構築する（フォールバック用）
        self._label_to_indices: dict[int, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            label = sample["label"]
            if label not in self._label_to_indices:
                self._label_to_indices[label] = []
            self._label_to_indices[label].append(idx)

    def __len__(self) -> int:
        """データセットのサンプル数を返す。"""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """指定インデックスのサンプルを返す。

        Returns:
            以下のキーを持つ辞書:
                - "token_ids": セグメントのトークンID列 (segment_len,)
                - "label": real=1, fake=0 のラベル (スカラ)
                - "utterance_id": 発話ID (文字列)
        """
        sample = self.samples[index]
        result = self._try_load_sample(sample)
        if result is not None:
            return result

        # 抽出に失敗した場合、同じラベルの別サンプルで最大3回リトライする
        label = sample["label"]
        candidates = [
            i for i in self._label_to_indices[label] if i != index
        ]
        fallback_indices = random.sample(
            candidates, min(3, len(candidates))
        )
        for fallback_idx in fallback_indices:
            fallback_sample = self.samples[fallback_idx]
            result = self._try_load_sample(fallback_sample)
            if result is not None:
                return result

        raise RuntimeError(
            f"セグメント抽出に失敗しました: index={index}, "
            f"utterance_id={sample['utterance_id']}, "
            f"フォールバックも全て失敗"
        )

    def _try_load_sample(self, sample: dict) -> Optional[dict]:
        """サンプルの読み込みとセグメント抽出を試みる。

        Args:
            sample: メタデータ辞書エントリ。

        Returns:
            成功時はサンプル辞書、失敗時はNone。
        """
        try:
            token_path = self.base_dir / sample["token_path"]
            token_ids = torch.load(token_path, weights_only=True)
            segment = self.segment_extractor.extract(token_ids)
            return {
                "token_ids": segment.long(),
                "label": torch.tensor(sample["label"], dtype=torch.float),
                "utterance_id": sample["utterance_id"],
            }
        except (ValueError, RuntimeError):
            return None

    @classmethod
    def create_dataloader(
        cls,
        dataset: "SpoofDetectionDataset",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DataLoader:
        """DataLoaderを作成する。

        Args:
            dataset: SpoofDetectionDatasetインスタンス。
            batch_size: バッチサイズ。
            shuffle: シャッフルするかどうか。
            num_workers: データ読み込みワーカー数。
            **kwargs: DataLoaderに渡す追加引数。

        Returns:
            設定済みのDataLoader。
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=cls._collate_fn,
            **kwargs,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        """バッチ内の辞書形式サンプルをまとめるcollate関数。

        Args:
            batch: __getitem__が返す辞書のリスト。

        Returns:
            以下のキーを持つ辞書:
                - "token_ids": (batch_size, segment_len)
                - "label": (batch_size,)
                - "utterance_id": 文字列のリスト
        """
        return {
            "token_ids": torch.stack([s["token_ids"] for s in batch]),
            "label": torch.stack([s["label"] for s in batch]),
            "utterance_id": [s["utterance_id"] for s in batch],
        }
