#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh
# Phase 2 データ準備パイプライン
#
# Stage 1: LibriTTS データセットのダウンロード
# Stage 2: 合成音声の生成とスプーフ検出器訓練用データセットの構築
#
# MSpoof-TTS のスプーフ検出器訓練に必要な一連のデータ準備を自動実行する。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# タイムスタンプ記録（総経過時間の計測用）
# ---------------------------------------------------------------------------
SCRIPT_START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# デフォルト設定変数
# ---------------------------------------------------------------------------

# Stage 1: ダウンロード関連
DATA_DIR="./data/libritts"                  # LibriTTS の保存先
SKIP_DOWNLOAD=false                         # ダウンロードをスキップするフラグ

# Stage 2: 合成データ生成・データセット構築関連
OUTPUT_DIR="./data/prepared"                # 前処理済みデータの出力先
NUM_SYNTHESIS=3                             # 各発話あたりの合成回数
BACKBONE="neu-tts-air"                      # TTS バックボーンモデル
CODEC="neucodec"                            # 音声コーデック
DEVICE="gpu"                                # 推論デバイス (gpu / cpu)
SEED_BASE=42                                # 乱数シードの基底値

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

# ステージヘッダーを表示する関数
print_stage_header() {
    local stage_num="$1"
    local stage_name="$2"
    info ""
    info "============================================================"
    info " Stage ${stage_num}: ${stage_name}"
    info " 開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
    info "============================================================"
    info ""
}

# 経過時間を人間が読みやすい形式で表示する関数
format_elapsed() {
    local elapsed=$1
    local hours=$((elapsed / 3600))
    local minutes=$(( (elapsed % 3600) / 60 ))
    local seconds=$((elapsed % 60))
    printf "%02d時間%02d分%02d秒" "${hours}" "${minutes}" "${seconds}"
}

# ---------------------------------------------------------------------------
# ヘルプメッセージの表示
# ---------------------------------------------------------------------------
usage() {
    cat <<HELP
使い方: $(basename "$0") [オプション]

Phase 2 データ準備パイプライン — LibriTTS ダウンロードから合成データ生成まで

オプション:
  --data-dir DIR        LibriTTS の保存先ディレクトリ (デフォルト: ${DATA_DIR})
  --output-dir DIR      前処理済みデータの出力先 (デフォルト: ${OUTPUT_DIR})
  --num-synthesis N     各発話あたりの合成回数 (デフォルト: ${NUM_SYNTHESIS})
  --backbone MODEL      TTS バックボーンモデル (デフォルト: ${BACKBONE})
  --codec CODEC         音声コーデック (デフォルト: ${CODEC})
  --device DEVICE       推論デバイス: gpu または cpu (デフォルト: ${DEVICE})
  --seed-base N         乱数シードの基底値 (デフォルト: ${SEED_BASE})
  --skip-download       Stage 1（ダウンロード）をスキップする
  --help, -h            このヘルプメッセージを表示する

例:
  # 全パイプラインを実行
  bash $(basename "$0")

  # ダウンロード済みの場合、合成データ生成のみ実行
  bash $(basename "$0") --skip-download --data-dir ./data/libritts

  # カスタム設定で実行
  bash $(basename "$0") --data-dir /mnt/data/libritts --output-dir /mnt/data/prepared --num-synthesis 5 --device cpu
HELP
    exit 0
}

# ---------------------------------------------------------------------------
# コマンドライン引数の解析
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-synthesis)
            NUM_SYNTHESIS="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --codec)
            CODEC="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed-base)
            SEED_BASE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            error "不明なオプション: $1（--help でヘルプを表示）"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# メイン処理の開始
# ---------------------------------------------------------------------------

info "============================================================"
info " MSpoof-TTS Phase 2 データ準備パイプライン"
info "============================================================"
info ""
info "設定:"
info "  DATA_DIR       = ${DATA_DIR}"
info "  OUTPUT_DIR     = ${OUTPUT_DIR}"
info "  NUM_SYNTHESIS  = ${NUM_SYNTHESIS}"
info "  BACKBONE       = ${BACKBONE}"
info "  CODEC          = ${CODEC}"
info "  DEVICE         = ${DEVICE}"
info "  SEED_BASE      = ${SEED_BASE}"
info "  SKIP_DOWNLOAD  = ${SKIP_DOWNLOAD}"
info ""

# 必要なコマンドの存在確認
check_command python

# =========================================================================
# Stage 1: LibriTTS データセットのダウンロード
# =========================================================================

if [[ "${SKIP_DOWNLOAD}" == true ]]; then
    info "Stage 1 をスキップします（--skip-download が指定されました）"
    info ""

    # スキップ時でもデータディレクトリの存在を確認する
    if [[ ! -d "${DATA_DIR}" ]]; then
        error "データディレクトリが見つかりません: ${DATA_DIR}"
    fi
else
    print_stage_header 1 "LibriTTS データセットのダウンロード"

    info "保存先ディレクトリ: ${DATA_DIR}"
    info "対象スプリット: ${SPLITS[*]}"
    info ""

    # 保存先ディレクトリを作成
    mkdir -p "${DATA_DIR}"

    # ダウンロード方法の自動選択:
    #   - HuggingFace datasets ライブラリが使えればそれを使う
    #   - なければ OpenSLR から wget/curl で直接ダウンロード
    if python -c "import datasets" 2>/dev/null; then
        info "HuggingFace datasets ライブラリを使用してダウンロードします"
        info ""

        for split in "${SPLITS[@]}"; do
            save_path="${DATA_DIR}/${split}"
            if [[ -d "${save_path}" ]]; then
                info "スプリットが既に存在します: ${save_path}（スキップ）"
                continue
            fi

            info "ダウンロード中: ${split} ..."
            python -c "
from datasets import load_dataset
import os

# HuggingFace からデータセットをダウンロード
ds = load_dataset('${HF_DATASET}', split='${split}')

# ローカルディスクに保存
save_path = '${save_path}'
os.makedirs(save_path, exist_ok=True)
ds.save_to_disk(save_path)

print(f'保存完了: {save_path} ({len(ds)} 件)')
"
            info "完了: ${split}"
        done
    else
        # OpenSLR から直接ダウンロードするフォールバック
        info "HuggingFace datasets ライブラリが見つかりません。"
        info "OpenSLR から直接ダウンロードします。"
        info ""
        info "注意: 合計で約 80GB 以上のディスク容量が必要です。"
        info ""

        # wget または curl の存在確認
        if command -v wget &>/dev/null; then
            DOWNLOADER="wget"
        elif command -v curl &>/dev/null; then
            DOWNLOADER="curl"
        else
            error "wget または curl が必要です。いずれかをインストールしてください。"
        fi

        for file in "${OPENSLR_FILES[@]}"; do
            url="${OPENSLR_BASE_URL}/${file}"
            dest="${DATA_DIR}/${file}"

            # ダウンロード済みファイルのスキップ判定
            if [[ -f "${dest}" ]]; then
                info "アーカイブが既に存在します: ${dest}（スキップ）"
            else
                info "ダウンロード中: ${url} ..."
                if [[ "${DOWNLOADER}" == "wget" ]]; then
                    wget -q --show-progress -O "${dest}" "${url}"
                else
                    curl -L --progress-bar -o "${dest}" "${url}"
                fi
                info "ダウンロード完了: ${file}"
            fi

            # tar.gz ファイルの展開
            info "展開中: ${file} ..."
            tar -xzf "${dest}" -C "${DATA_DIR}"
            info "展開完了: ${file}"
        done
    fi

    # Stage 1 完了 — ディレクトリ内のファイル数を表示
    info ""
    if command -v find &>/dev/null; then
        total_files=$(find "${DATA_DIR}" -type f | wc -l | tr -d ' ')
        info "Stage 1 完了 — 合計ファイル数: ${total_files}"
    else
        info "Stage 1 完了"
    fi
fi

# =========================================================================
# Stage 2: 合成データの生成とデータセット構築
# =========================================================================

print_stage_header 2 "合成データの生成とデータセット構築"

info "入力ディレクトリ:     ${DATA_DIR}"
info "出力ディレクトリ:     ${OUTPUT_DIR}"
info "合成回数/発話:        ${NUM_SYNTHESIS}"
info "バックボーンモデル:   ${BACKBONE}"
info "コーデック:           ${CODEC}"
info "デバイス:             ${DEVICE}"
info "シードベース:         ${SEED_BASE}"
info ""

# 出力ディレクトリを作成
mkdir -p "${OUTPUT_DIR}"

# Python の前処理スクリプトを実行
info "mspoof_tts.data.prepare を実行します ..."
info ""

python -m mspoof_tts.data.prepare \
    --libritts_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_synthesis "${NUM_SYNTHESIS}" \
    --backbone "${BACKBONE}" \
    --codec "${CODEC}" \
    --device "${DEVICE}" \
    --seed_base "${SEED_BASE}" \
    --resume

info ""
info "Stage 2 完了"

# =========================================================================
# 完了メッセージと総経過時間の表示
# =========================================================================

SCRIPT_END_TIME=$(date +%s)
ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

info ""
info "============================================================"
info " データ準備パイプライン完了"
info "============================================================"
info ""
info "出力ディレクトリ: ${OUTPUT_DIR}"
info "総経過時間:       $(format_elapsed ${ELAPSED})"
info ""
info "次のステップ:"
info "  scripts/train_detectors.sh でスプーフ検出器を訓練する"
info ""
