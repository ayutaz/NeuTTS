"""データ処理モジュール。

スプーフ検出器の訓練に必要なデータセット構築、
セグメント生成、合成データ準備のためのモジュールを含む。
"""

from mspoof_tts.data.dataset import SpoofDetectionDataset
from mspoof_tts.data.segment import SegmentExtractor

__all__ = ["SpoofDetectionDataset", "SegmentExtractor"]
