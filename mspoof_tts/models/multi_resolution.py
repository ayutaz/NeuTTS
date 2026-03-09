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

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from mspoof_tts.data.segment import SegmentExtractor, SegmentMode
from mspoof_tts.models.spoof_detector import SpoofDetector

logger = logging.getLogger(__name__)


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
        self.detectors = nn.ModuleDict({
            name: SpoofDetector(vocab_size, d_model, n_heads, n_layers, ffn_dim, dropout)
            for name in self.DETECTOR_NAMES
        })
        # 推論時は random_offset=False で末尾から決定的に抽出する
        self.extractors: dict[str, SegmentExtractor] = {
            "M_10": SegmentExtractor(SegmentMode.CONTIGUOUS_10, random_offset=False),
            "M_25": SegmentExtractor(SegmentMode.CONTIGUOUS_25, random_offset=False),
            "M_50": SegmentExtractor(SegmentMode.CONTIGUOUS_50, random_offset=False),
            "M_50_25": SegmentExtractor(SegmentMode.SKIP_50_25, random_offset=False),
            "M_50_10": SegmentExtractor(SegmentMode.SKIP_50_10, random_offset=False),
        }

    def load_checkpoints(self, checkpoint_dir: Path) -> None:
        """各検出器のチェックポイントをディレクトリからロードする。

        Args:
            checkpoint_dir: チェックポイントファイルが格納されたディレクトリ。
                各ファイルは {detector_name}.pt の形式を期待する。
        """
        checkpoint_dir = Path(checkpoint_dir)
        for name in self.DETECTOR_NAMES:
            ckpt_path = checkpoint_dir / f"{name}.pt"
            if not ckpt_path.exists():
                logger.warning("チェックポイントが見つかりません: %s", ckpt_path)
                continue
            state = torch.load(ckpt_path, weights_only=True)
            # チェックポイントが辞書形式で model_state_dict キーを持つ場合に対応
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.detectors[name].load_state_dict(state)
            logger.info("検出器 %s のチェックポイントをロードしました: %s", name, ckpt_path)

    def score_short(self, token_ids: torch.Tensor) -> torch.Tensor:
        """短スパン検出器 M_10 でスコアリングする (Stage 1 枝刈り用)。

        Args:
            token_ids: トークンID列 (batch, seq_len)。seq_len >= 10。

        Returns:
            realスコア (batch,)。
        """
        # 末尾10トークンを切り出して M_10 に入力
        segment = token_ids[:, -10:]
        return self.detectors["M_10"](segment)

    def score_mid(self, token_ids: torch.Tensor) -> torch.Tensor:
        """中距離検出器 M_25 でスコアリングする (Stage 2 枝刈り用)。

        Args:
            token_ids: トークンID列 (batch, seq_len)。seq_len >= 25。

        Returns:
            realスコア (batch,)。
        """
        # 末尾25トークンを切り出して M_25 に入力
        segment = token_ids[:, -25:]
        return self.detectors["M_25"](segment)

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
        # 末尾50トークンを取得
        last_50 = token_ids[:, -50:]

        # M_50: 連続50トークンをそのまま使用
        score_m50 = self.detectors["M_50"](last_50)

        # M_50_25: 50トークンからスキップ率 r=2 でサンプリング (25トークン)
        skip_25 = last_50[:, ::2]
        score_m50_25 = self.detectors["M_50_25"](skip_25)

        # M_50_10: 50トークンからスキップ率 r=5 でサンプリング (10トークン)
        skip_10 = last_50[:, ::5]
        score_m50_10 = self.detectors["M_50_10"](skip_10)

        return {
            "M_50": score_m50,
            "M_50_25": score_m50_25,
            "M_50_10": score_m50_10,
        }

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
        if weights is None:
            weights = {"M_50": 1.0, "M_50_25": 1.0, "M_50_10": 1.0}

        total_rank: Optional[torch.Tensor] = None
        for name, s in scores.items():
            # argsort(-s) で降順のインデックスを取得し、
            # 再度 argsort することで各要素のランクを得る (0-indexed)
            # スコアが高いほどランクが小さい (良い)
            ranks = torch.argsort(torch.argsort(-s)).float()
            w = weights[name]
            if total_rank is None:
                total_rank = w * ranks
            else:
                total_rank = total_rank + w * ranks

        # 総合ランクが最も小さい候補のインデックスを返す
        return torch.argmin(total_rank)  # type: ignore[arg-type]
