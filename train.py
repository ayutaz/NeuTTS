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
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from mspoof_tts.data.dataset import SpoofDetectionDataset
from mspoof_tts.data.segment import SegmentExtractor, SegmentMode
from mspoof_tts.models.spoof_detector import SpoofDetector

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
    # 1. セットアップ: 乱数シードとデバイス
    # -----------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("デバイス: %s", device)

    DETECTOR_TO_MODE = {
        "M_10": SegmentMode.CONTIGUOUS_10,
        "M_25": SegmentMode.CONTIGUOUS_25,
        "M_50": SegmentMode.CONTIGUOUS_50,
        "M_50_25": SegmentMode.SKIP_50_25,
        "M_50_10": SegmentMode.SKIP_50_10,
    }
    segment_mode = DETECTOR_TO_MODE[args.detector_name]

    # -----------------------------------------------------------------------
    # 2. データセットの構築
    # -----------------------------------------------------------------------
    logger.info("データセットを構築しています...")

    segment_extractor = SegmentExtractor(mode=segment_mode, random_offset=True)
    metadata_path = Path(args.data_dir) / "metadata.jsonl"

    train_dataset = SpoofDetectionDataset(
        metadata_path=metadata_path,
        segment_extractor=segment_extractor,
        split="train",
    )
    val_dataset = SpoofDetectionDataset(
        metadata_path=metadata_path,
        segment_extractor=segment_extractor,
        split="val",
    )

    train_loader = SpoofDetectionDataset.create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = SpoofDetectionDataset.create_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info("訓練データ: %d サンプル", len(train_dataset))
    logger.info("検証データ: %d サンプル", len(val_dataset))

    # -----------------------------------------------------------------------
    # 3. モデルの構築
    # -----------------------------------------------------------------------
    logger.info("モデルを構築しています...")

    model = SpoofDetector(
        vocab_size=65536,
        d_model=args.d_model,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("パラメータ数: %d (訓練可能: %d)", num_params, num_trainable)

    # -----------------------------------------------------------------------
    # 4. オプティマイザ・損失関数・スケジューラ
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # -----------------------------------------------------------------------
    # 5. チェックポイントからの再開
    # -----------------------------------------------------------------------
    start_epoch = 0
    if args.resume is not None:
        logger.info("チェックポイントから再開: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info("エポック %d から再開します", start_epoch)

    # -----------------------------------------------------------------------
    # 6. 訓練ループ
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = 0
    checkpoint_dir = Path(args.checkpoint_dir)

    logger.info("訓練を開始します...")
    logger.info("")

    for epoch in range(start_epoch, args.epochs):
        # -------------------------------------------------------------------
        # 6a. 訓練フェーズ
        # -------------------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"エポック {epoch + 1}/{args.epochs} [訓練]",
            leave=False,
        )
        for batch in train_pbar:
            token_ids = batch["token_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            predictions = model(token_ids)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            predicted_labels = (predictions >= 0.5).float()
            train_correct += (predicted_labels == labels).sum().item()
            train_total += batch_size

            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{train_correct / train_total:.4f}",
            )

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # -------------------------------------------------------------------
        # 6b. 検証フェーズ
        # -------------------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(
            val_loader,
            desc=f"エポック {epoch + 1}/{args.epochs} [検証]",
            leave=False,
        )
        with torch.no_grad():
            for batch in val_pbar:
                token_ids = batch["token_ids"].to(device)
                labels = batch["label"].to(device)

                predictions = model(token_ids)
                loss = criterion(predictions, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_total += batch_size

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / val_total
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        predicted_binary = (all_preds_np >= 0.5).astype(int)
        labels_binary = all_labels_np.astype(int)

        val_acc = accuracy_score(labels_binary, predicted_binary)
        val_f1 = f1_score(labels_binary, predicted_binary, average="macro")
        try:
            val_auc = roc_auc_score(labels_binary, all_preds_np)
        except ValueError:
            # 全ラベルが同一クラスの場合、AUCは計算不可
            val_auc = 0.0
            logger.warning("AUCを計算できません（ラベルが単一クラスの可能性）")

        # スケジューラを更新
        scheduler.step()

        # -------------------------------------------------------------------
        # 6c. ログ出力
        # -------------------------------------------------------------------
        logger.info(
            "エポック %d/%d | "
            "訓練損失: %.4f, 訓練精度: %.4f | "
            "検証損失: %.4f, 検証精度: %.4f, 検証F1: %.4f, 検証AUC: %.4f | "
            "学習率: %.2e",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
            val_auc,
            optimizer.param_groups[0]["lr"],
        )

        # -------------------------------------------------------------------
        # 6d. チェックポイントの保存
        # -------------------------------------------------------------------
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_auc,
        }

        # 定期保存
        if (epoch + 1) % args.save_every == 0:
            path = checkpoint_dir / f"{args.detector_name}_epoch{epoch + 1}.pt"
            torch.save(checkpoint_data, path)
            logger.info("チェックポイントを保存しました: %s", path)

        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_path = checkpoint_dir / f"{args.detector_name}_best.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(
                "ベストモデルを更新しました (エポック %d, 検証損失: %.4f, 検証AUC: %.4f): %s",
                best_epoch,
                best_val_loss,
                best_val_auc,
                best_path,
            )

    # -----------------------------------------------------------------------
    # 7. 最終サマリーとチェックポイントの保存
    # -----------------------------------------------------------------------
    final_path = checkpoint_dir / f"{args.detector_name}.pt"
    final_checkpoint = {
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_auc": val_auc,
    }
    torch.save(final_checkpoint, final_path)

    logger.info("")
    logger.info("========================================")
    logger.info("訓練完了")
    logger.info("========================================")
    logger.info("ベストエポック:   %d", best_epoch)
    logger.info("ベスト検証損失:   %.4f", best_val_loss)
    logger.info("ベスト検証AUC:    %.4f", best_val_auc)
    logger.info("最終モデル保存先: %s", final_path)
    logger.info("ベストモデル保存先: %s", checkpoint_dir / f"{args.detector_name}_best.pt")


if __name__ == "__main__":
    main()
