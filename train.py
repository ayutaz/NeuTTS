#!/usr/bin/env python3
"""
train.py - MSpoof-TTS スプーフ検出器の訓練エントリポイント

Conformer ベースのマルチ解像度スプーフ検出器を訓練する。
各検出器は、リアル（実音声由来）とフェイク（合成音声由来）の
離散コーデックトークンセグメントを二値分類する。

使い方:
    python train.py \
        --detector-name M_50 \
        --segment-mode contiguous \
        --segment-length 50 \
        --data-dir ./data/libritts \
        --checkpoint-dir ./checkpoints/M_50 \
        --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(
        description="MSpoof-TTS スプーフ検出器の訓練",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -----------------------------------------------------------------------
    # 検出器の設定
    # -----------------------------------------------------------------------
    detector_group = parser.add_argument_group("検出器の設定")

    detector_group.add_argument(
        "--detector-name",
        type=str,
        required=True,
        choices=["M_50", "M_25", "M_10", "M_50_25", "M_50_10"],
        help="訓練する検出器の名前",
    )
    detector_group.add_argument(
        "--segment-mode",
        type=str,
        required=True,
        choices=["contiguous", "skip"],
        help="セグメント構築方法（contiguous: 連続クロッピング, skip: スキップサンプリング）",
    )
    detector_group.add_argument(
        "--segment-length",
        type=int,
        required=True,
        help="モデルに入力するセグメントのトークン数",
    )
    detector_group.add_argument(
        "--span-length",
        type=int,
        default=0,
        help="スキップサンプリング時のスパン長（連続クロッピングの場合は0）",
    )
    detector_group.add_argument(
        "--skip-rate",
        type=int,
        default=0,
        help="スキップサンプリング時のスキップ率 r（連続クロッピングの場合は0）",
    )

    # -----------------------------------------------------------------------
    # モデルアーキテクチャの設定
    # -----------------------------------------------------------------------
    model_group = parser.add_argument_group("モデルアーキテクチャの設定")

    model_group.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Conformer のモデル次元 d_model",
    )
    model_group.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="マルチヘッドアテンションのヘッド数",
    )
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Conformer 層の数",
    )
    model_group.add_argument(
        "--ffn-dim",
        type=int,
        default=1024,
        help="フィードフォワードネットワークの次元",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="ドロップアウト率",
    )

    # -----------------------------------------------------------------------
    # データの設定
    # -----------------------------------------------------------------------
    data_group = parser.add_argument_group("データの設定")

    data_group.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="訓練データのルートディレクトリ",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="検証データの割合",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="データローダーのワーカー数",
    )

    # -----------------------------------------------------------------------
    # 訓練ハイパーパラメータ
    # -----------------------------------------------------------------------
    train_group = parser.add_argument_group("訓練ハイパーパラメータ")

    train_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="バッチサイズ",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="訓練エポック数",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="学習率",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )

    # -----------------------------------------------------------------------
    # 出力の設定
    # -----------------------------------------------------------------------
    output_group = parser.add_argument_group("出力の設定")

    output_group.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="チェックポイントの保存先ディレクトリ",
    )
    output_group.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="ログの保存先ディレクトリ",
    )
    output_group.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="チェックポイントの保存間隔（エポック数）",
    )

    # -----------------------------------------------------------------------
    # その他の設定
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（再現性のため）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="訓練を再開するチェックポイントのパス",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """引数の整合性を検証する。"""

    # スキップサンプリングモードの場合、スパン長とスキップ率が必要
    if args.segment_mode == "skip":
        if args.span_length <= 0:
            raise ValueError(
                "スキップサンプリングモードでは --span-length を正の値で指定してください"
            )
        if args.skip_rate <= 0:
            raise ValueError(
                "スキップサンプリングモードでは --skip-rate を正の値で指定してください"
            )
        # スパン長をスキップ率で割った結果がセグメント長と一致することを確認
        expected_length = args.span_length // args.skip_rate
        if expected_length != args.segment_length:
            raise ValueError(
                f"スパン長({args.span_length}) / スキップ率({args.skip_rate}) = "
                f"{expected_length} がセグメント長({args.segment_length})と一致しません"
            )

    # データディレクトリの存在確認
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"データディレクトリが見つかりません: {args.data_dir}")

    # モデルパラメータの妥当性検証
    if args.d_model % args.num_heads != 0:
        raise ValueError(
            f"d_model({args.d_model}) は num_heads({args.num_heads}) で割り切れる必要があります"
        )


def main() -> None:
    """訓練のメインエントリポイント。"""

    args = parse_args()

    logger.info("========================================")
    logger.info("MSpoof-TTS スプーフ検出器の訓練")
    logger.info("========================================")
    logger.info("")
    logger.info("検出器の設定:")
    logger.info("  検出器名:       %s", args.detector_name)
    logger.info("  セグメント方式: %s", args.segment_mode)
    logger.info("  セグメント長:   %d", args.segment_length)
    if args.segment_mode == "skip":
        logger.info("  スパン長:       %d", args.span_length)
        logger.info("  スキップ率:     %d", args.skip_rate)
    logger.info("")
    logger.info("モデルアーキテクチャ:")
    logger.info("  d_model:   %d", args.d_model)
    logger.info("  num_heads: %d", args.num_heads)
    logger.info("  num_layers: %d", args.num_layers)
    logger.info("  ffn_dim:   %d", args.ffn_dim)
    logger.info("  dropout:   %.2f", args.dropout)
    logger.info("")
    logger.info("訓練ハイパーパラメータ:")
    logger.info("  バッチサイズ: %d", args.batch_size)
    logger.info("  エポック数:   %d", args.epochs)
    logger.info("  学習率:       %s", args.learning_rate)
    logger.info("  Weight decay: %s", args.weight_decay)
    logger.info("")

    # 引数の整合性を検証
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error("引数の検証に失敗しました: %s", e)
        sys.exit(1)

    # チェックポイントとログのディレクトリを作成
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # TODO: 以下の訓練パイプラインを実装する
    # -----------------------------------------------------------------------
    #
    # 1. データセットの構築
    #    - mspoof_tts.data.dataset からリアル/フェイクのトークンペアを読み込む
    #    - mspoof_tts.data.segment でセグメントを構築する
    #    - 訓練/検証に分割する
    #
    # 2. モデルの構築
    #    - mspoof_tts.models.spoof_detector から Conformer ベースの検出器を構築する
    #    - Embedding Layer → Conformer Block x4 → Adaptive Pooling → Classifier Head
    #
    # 3. 訓練ループ
    #    - AdamW オプティマイザ（lr=1e-4, weight_decay=1e-4）
    #    - Binary Cross-Entropy (BCE) 損失関数
    #    - エポックごとに検証セットで評価（AUC, Accuracy, Macro-F1）
    #    - save_every エポックごとにチェックポイントを保存
    #
    # 4. 最終チェックポイントの保存
    #    - ベストモデル（検証損失が最小）のチェックポイントを保存する
    #

    logger.info("TODO: 訓練パイプラインの実装が必要です")
    logger.info("詳細は docs/05_training.md を参照してください")
    logger.info("")
    logger.info("実装予定のステップ:")
    logger.info("  1. データセット構築（mspoof_tts.data.dataset）")
    logger.info("  2. セグメント構築（mspoof_tts.data.segment）")
    logger.info("  3. Conformer 検出器の構築（mspoof_tts.models.spoof_detector）")
    logger.info("  4. 訓練ループ（AdamW + BCE）")
    logger.info("  5. チェックポイントの保存")


if __name__ == "__main__":
    main()
