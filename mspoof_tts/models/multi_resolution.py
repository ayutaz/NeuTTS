"""マルチ解像度スプーフ検出器の管理モジュール。

5つの検出器を統合的に管理し、階層的デコーディングの各ステージで
適切な検出器を提供する。

検出器一覧:
    - M_10: 連続10トークンセグメント (contiguous cropping, L=10)
    - M_25: 連続25トークンセグメント (contiguous cropping, L=25)
    - M_50: 連続50トークンセグメント (contiguous cropping, L=50)
    - M_50_25: 50トークンスパンからスキップサンプリング (r=2, 25トークン選択)
    - M_50_10: 50トークンスパンからスキップサンプリング (r=5, 10トークン選択)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from mspoof_tts.models.spoof_detector import SpoofDetector


class MultiResolutionDetector(nn.Module):
    """5つのマルチ解像度スプーフ検出器を統合管理するクラス。

    各検出器は同一アーキテクチャ・異なるパラメータを持ち、
    異なる時間解像度でのスプーフ検出を担当する。

    Args:
        vocab_size: コーデックトークンの語彙サイズ。
        d_model: モデルの埋め込み次元数。
        n_heads: マルチヘッドアテンションのヘッド数。
        n_layers: Conformerブロックの積み重ね数。
        ffn_dim: フィードフォワード層の内部次元数。
        dropout: ドロップアウト率。
    """

    # 検出器名のリスト
    DETECTOR_NAMES: list[str] = ["M_10", "M_25", "M_50", "M_50_25", "M_50_10"]

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("MultiResolutionDetector の実装が必要")

    def load_checkpoints(self, checkpoint_dir: Path) -> None:
        """各検出器のチェックポイントをディレクトリからロードする。

        Args:
            checkpoint_dir: チェックポイントファイルが格納されたディレクトリ。
                各ファイルは {detector_name}.pt の形式を期待する。
        """
        raise NotImplementedError

    def score_short(self, token_ids: torch.Tensor) -> torch.Tensor:
        """短スパン検出器 M_10 でスコアリングする (Stage 1 枝刈り用)。

        Args:
            token_ids: トークンID列 (batch, seq_len)。seq_len >= 10。

        Returns:
            realスコア (batch,)。
        """
        raise NotImplementedError

    def score_mid(self, token_ids: torch.Tensor) -> torch.Tensor:
        """中距離検出器 M_25 でスコアリングする (Stage 2 枝刈り用)。

        Args:
            token_ids: トークンID列 (batch, seq_len)。seq_len >= 25。

        Returns:
            realスコア (batch,)。
        """
        raise NotImplementedError

    def score_long(
        self, token_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """長スパン検出器3つ (M_50, M_50_25, M_50_10) でスコアリングする (最終選択用)。

        マルチ解像度ランク集約のため、3つの検出器のスコアを辞書で返す。

        Args:
            token_ids: トークンID列 (batch, seq_len)。seq_len >= 50。

        Returns:
            各検出器名をキーとするスコア辞書。
        """
        raise NotImplementedError

    def rank_aggregate(
        self,
        scores: dict[str, torch.Tensor],
        weights: Optional[dict[str, float]] = None,
    ) -> torch.Tensor:
        """マルチ解像度ランク集約を行い、最良候補のインデックスを返す。

        各検出器のスコアをランクに変換し、重み付き和で総合ランクを算出する。

        Args:
            scores: score_long() の出力。
            weights: 各検出器のランク重み。Noneの場合は均等重み。

        Returns:
            最良候補のインデックス (スカラ)。
        """
        raise NotImplementedError
