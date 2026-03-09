#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh
# LibriTTS データセットのダウンロードと前処理を行うスクリプト
#
# MSpoof-TTS のスプーフ検出器訓練に必要な LibriTTS データセットを取得する。
# train-clean-100, train-clean-360, train-other-500 の3つのスプリットを対象とする。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 設定変数（必要に応じて変更してください）
# ---------------------------------------------------------------------------

# データの保存先ディレクトリ
DATA_DIR="${DATA_DIR:-./data/libritts}"

# HuggingFace のデータセット名
HF_DATASET="openslr/libri-tts"

# ダウンロード対象のスプリット一覧
SPLITS=(
    "train.clean.100"
    "train.clean.360"
    "train.other.500"
)

# OpenSLR からの直接ダウンロード URL（HuggingFace が使えない場合の代替手段）
OPENSLR_BASE_URL="https://www.openslr.org/resources/60"
OPENSLR_FILES=(
    "train-clean-100.tar.gz"
    "train-clean-360.tar.gz"
    "train-other-500.tar.gz"
)

# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

# 情報メッセージを表示する関数
info() {
    echo "[情報] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

# エラーメッセージを表示して終了する関数
error() {
    echo "[エラー] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    exit 1
}

# 警告メッセージを表示する関数
warn() {
    echo "[警告] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

# コマンドの存在を確認する関数
check_command() {
    if ! command -v "$1" &>/dev/null; then
        error "必要なコマンド '$1' が見つかりません。インストールしてください。"
    fi
}

# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

info "=========================================="
info "LibriTTS データセット準備スクリプト"
info "=========================================="
info ""
info "保存先ディレクトリ: ${DATA_DIR}"
info "対象スプリット: ${SPLITS[*]}"
info ""

# 必要なコマンドの存在確認
check_command python3

# ダウンロード方法の選択
info "ダウンロード方法を選択してください:"
info "  1) HuggingFace datasets ライブラリを使用（推奨）"
info "  2) OpenSLR から直接ダウンロード（wget/curl使用）"
info ""
read -rp "選択 [1/2]: " DOWNLOAD_METHOD

case "${DOWNLOAD_METHOD}" in
    1)
        # -------------------------------------------------------------------
        # 方法1: HuggingFace datasets を使用してダウンロード
        # -------------------------------------------------------------------

        # HuggingFace datasets ライブラリの存在確認
        if ! python3 -c "import datasets" 2>/dev/null; then
            warn "HuggingFace datasets ライブラリが見つかりません。"
            read -rp "pip install datasets でインストールしますか？ [y/N]: " INSTALL_HF
            if [[ "${INSTALL_HF}" =~ ^[Yy]$ ]]; then
                pip install datasets
            else
                error "HuggingFace datasets ライブラリが必要です。"
            fi
        fi

        # ダウンロードの確認プロンプト
        info ""
        info "以下のコマンドが実行されます:"
        info ""
        for split in "${SPLITS[@]}"; do
            info "  python3 -c \\"
            info "    \"from datasets import load_dataset; \\"
            info "     ds = load_dataset('${HF_DATASET}', split='${split}'); \\"
            info "     ds.save_to_disk('${DATA_DIR}/${split}')\""
            info ""
        done

        # ユーザーに確認を求める
        read -rp "ダウンロードを開始しますか？ [y/N]: " CONFIRM
        if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
            info "ダウンロードをキャンセルしました。"
            exit 0
        fi

        # 保存先ディレクトリを作成
        mkdir -p "${DATA_DIR}"

        # 各スプリットをダウンロード
        for split in "${SPLITS[@]}"; do
            info "ダウンロード中: ${split} ..."
            python3 -c "
from datasets import load_dataset
import os

# HuggingFace からデータセットをダウンロード
ds = load_dataset('${HF_DATASET}', split='${split}')

# ローカルディスクに保存
save_path = os.path.join('${DATA_DIR}', '${split}')
os.makedirs(save_path, exist_ok=True)
ds.save_to_disk(save_path)

print(f'保存完了: {save_path} ({len(ds)} 件)')
"
            info "完了: ${split}"
        done
        ;;

    2)
        # -------------------------------------------------------------------
        # 方法2: OpenSLR から直接ダウンロードして展開
        # -------------------------------------------------------------------

        # wget または curl の存在確認
        if command -v wget &>/dev/null; then
            DOWNLOADER="wget"
        elif command -v curl &>/dev/null; then
            DOWNLOADER="curl"
        else
            error "wget または curl が必要です。いずれかをインストールしてください。"
        fi

        # ダウンロードの確認プロンプト
        info ""
        info "以下のファイルをダウンロードして展開します:"
        info ""
        for file in "${OPENSLR_FILES[@]}"; do
            info "  ${OPENSLR_BASE_URL}/${file}"
        done
        info ""
        info "展開先: ${DATA_DIR}"
        info ""
        info "注意: 合計で約 80GB 以上のディスク容量が必要です。"
        info ""

        # ユーザーに確認を求める
        read -rp "ダウンロードを開始しますか？ [y/N]: " CONFIRM
        if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
            info "ダウンロードをキャンセルしました。"
            exit 0
        fi

        # 保存先ディレクトリを作成
        mkdir -p "${DATA_DIR}"

        # 各ファイルをダウンロードして展開
        for file in "${OPENSLR_FILES[@]}"; do
            url="${OPENSLR_BASE_URL}/${file}"
            dest="${DATA_DIR}/${file}"

            # ダウンロード済みファイルのスキップ判定
            if [[ -f "${dest}" ]]; then
                warn "ファイルが既に存在します: ${dest} （スキップ）"
            else
                info "ダウンロード中: ${url} ..."
                if [[ "${DOWNLOADER}" == "wget" ]]; then
                    wget -O "${dest}" "${url}"
                else
                    curl -L -o "${dest}" "${url}"
                fi
                info "ダウンロード完了: ${file}"
            fi

            # tar.gz ファイルの展開
            info "展開中: ${file} ..."
            tar -xzf "${dest}" -C "${DATA_DIR}"
            info "展開完了: ${file}"
        done
        ;;

    *)
        error "無効な選択です。1 または 2 を入力してください。"
        ;;
esac

# ---------------------------------------------------------------------------
# データセット構造の確認
# ---------------------------------------------------------------------------
info ""
info "=========================================="
info "ダウンロード完了"
info "=========================================="
info "データディレクトリ: ${DATA_DIR}"
info ""

# ディレクトリ内のファイル数を表示
if command -v find &>/dev/null; then
    total_files=$(find "${DATA_DIR}" -type f | wc -l | tr -d ' ')
    info "合計ファイル数: ${total_files}"
fi

info ""
info "次のステップ:"
info "  1. NeuTTS で合成音声を生成する"
info "  2. NeuCodec で離散トークン列に変換する"
info "  3. scripts/train_detectors.sh でスプーフ検出器を訓練する"
info ""
