#!/usr/bin/env bash
# =============================================================================
# train_detectors.sh
# 5つのマルチ解像度スプーフ検出器を順番に訓練するスクリプト
#
# 各検出器は独立したパラメータを持つ Conformer ベースの二値分類器であり、
# 異なる時間解像度でリアル/フェイクのトークンセグメントを識別する。
#
# 使い方:
#   bash scripts/train_detectors.sh
#   bash scripts/train_detectors.sh --epochs 50 --batch-size 32
#   bash scripts/train_detectors.sh --detectors M_50,M_25 --gpu 1
#   bash scripts/train_detectors.sh --help
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値の定義
# ---------------------------------------------------------------------------
DATA_DIR="./data/prepared"
CHECKPOINT_DIR="./checkpoints/detectors"
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE="1e-4"
GPU_ID=0
DETECTORS_CSV=""

# 全検出器の一覧（デフォルト）
ALL_DETECTORS="M_50,M_25,M_10,M_50_25,M_50_10"

# ---------------------------------------------------------------------------
# ヘルプメッセージの表示
# ---------------------------------------------------------------------------
usage() {
    cat <<USAGE
使い方: $(basename "$0") [オプション]

5つのマルチ解像度スプーフ検出器を順番に訓練する。

オプション:
  --data-dir DIR        訓練データのディレクトリ         (デフォルト: ${DATA_DIR})
  --checkpoint-dir DIR  チェックポイントの保存先         (デフォルト: ${CHECKPOINT_DIR})
  --epochs N            訓練エポック数                   (デフォルト: ${EPOCHS})
  --batch-size N        バッチサイズ                     (デフォルト: ${BATCH_SIZE})
  --learning-rate LR    学習率                           (デフォルト: ${LEARNING_RATE})
  --gpu ID              使用する GPU の ID               (デフォルト: ${GPU_ID})
  --detectors LIST      訓練する検出器（カンマ区切り）   (デフォルト: 全5検出器)
                        選択可能: M_50, M_25, M_10, M_50_25, M_50_10
  --help                このヘルプメッセージを表示する

例:
  # 全検出器を訓練する
  $(basename "$0")

  # 特定の検出器のみ訓練する
  $(basename "$0") --detectors M_50,M_25

  # GPU 1 でバッチサイズ 32 で訓練する
  $(basename "$0") --gpu 1 --batch-size 32
USAGE
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
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --detectors)
            DETECTORS_CSV="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "[エラー] 不明なオプション: $1" >&2
            echo "ヘルプを表示するには --help を指定してください。" >&2
            exit 1
            ;;
    esac
done

# 検出器リストが未指定の場合は全検出器を使用する
if [[ -z "${DETECTORS_CSV}" ]]; then
    DETECTORS_CSV="${ALL_DETECTORS}"
fi

# ---------------------------------------------------------------------------
# 検出器ごとの訓練パラメータを定義する関数
#
# 各検出器の segment-mode, segment-length, span-length, skip-rate を返す
# ---------------------------------------------------------------------------
get_detector_params() {
    local name="$1"
    case "${name}" in
        M_50)
            # 長さ50の連続部分列で訓練
            echo "contiguous 50 0 0"
            ;;
        M_25)
            # 長さ25の連続部分列で訓練
            echo "contiguous 25 0 0"
            ;;
        M_10)
            # 長さ10の連続部分列で訓練
            echo "contiguous 10 0 0"
            ;;
        M_50_25)
            # 50トークンスパンからr=2でスキップサンプリング（25トークン）
            echo "skip 25 50 2"
            ;;
        M_50_10)
            # 50トークンスパンからr=5でスキップサンプリング（10トークン）
            echo "skip 10 50 5"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

# 情報メッセージを表示する
info() {
    echo "[情報] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

# 警告メッセージを表示する
warn() {
    echo "[警告] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
}

# エラーメッセージを表示して終了する
error() {
    echo "[エラー] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    exit 1
}

# 経過時間を人間が読みやすい形式で表示する
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d時間%02d分%02d秒" "${hours}" "${minutes}" "${secs}"
}

# ---------------------------------------------------------------------------
# 検出器リストの検証
# ---------------------------------------------------------------------------
IFS=',' read -ra SELECTED_DETECTORS <<< "${DETECTORS_CSV}"

for name in "${SELECTED_DETECTORS[@]}"; do
    params=$(get_detector_params "${name}")
    if [[ -z "${params}" ]]; then
        error "不明な検出器名: ${name}（選択可能: M_50, M_25, M_10, M_50_25, M_50_10）"
    fi
done

DETECTOR_COUNT=${#SELECTED_DETECTORS[@]}

# ---------------------------------------------------------------------------
# train.py の存在確認
# ---------------------------------------------------------------------------
TRAIN_SCRIPT="./train.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    error "訓練スクリプトが見つかりません: ${TRAIN_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# uv コマンドの存在確認
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    error "uv コマンドが見つかりません。uv をインストールしてください: https://docs.astral.sh/uv/"
fi

# ---------------------------------------------------------------------------
# メイン処理の開始
# ---------------------------------------------------------------------------
info "=========================================="
info "マルチ解像度スプーフ検出器 訓練スクリプト"
info "=========================================="
info ""
info "設定:"
info "  GPU ID:          ${GPU_ID}"
info "  バッチサイズ:    ${BATCH_SIZE}"
info "  エポック数:      ${EPOCHS}"
info "  学習率:          ${LEARNING_RATE}"
info "  データ:          ${DATA_DIR}"
info "  チェックポイント: ${CHECKPOINT_DIR}"
info "  対象検出器:      ${DETECTORS_CSV} (${DETECTOR_COUNT}個)"
info ""

# ディレクトリの作成
mkdir -p "${CHECKPOINT_DIR}"

# GPU の設定
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
info ""

# 全体の開始時刻を記録
TOTAL_START_TIME=$(date +%s)

# 訓練の成功/失敗を記録する配列
declare -a RESULTS=()
declare -a DURATIONS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

# ---------------------------------------------------------------------------
# 選択された検出器を順番に訓練する
# ---------------------------------------------------------------------------
for i in "${!SELECTED_DETECTORS[@]}"; do
    name="${SELECTED_DETECTORS[$i]}"

    # 検出器固有のパラメータを取得する
    read -r seg_mode seg_len span_len skip_rate <<< "$(get_detector_params "${name}")"

    info "=========================================="
    info "検出器 $((i + 1))/${DETECTOR_COUNT}: ${name} の訓練を開始"
    info "  セグメント方式: ${seg_mode}"
    info "  セグメント長:   ${seg_len}"
    if [[ "${seg_mode}" == "skip" ]]; then
        info "  スパン長:       ${span_len}"
        info "  スキップ率:     ${skip_rate}"
    fi
    info "=========================================="

    # 検出器ごとのチェックポイントとログディレクトリ
    detector_ckpt_dir="${CHECKPOINT_DIR}/${name}"
    detector_log_dir="${CHECKPOINT_DIR}/${name}/logs"
    mkdir -p "${detector_ckpt_dir}"
    mkdir -p "${detector_log_dir}"

    # 訓練開始時刻を記録
    START_TIME=$(date +%s)

    # 訓練コマンドの構築
    # 連続クロッピングの場合は --span-length と --skip-rate を渡さない
    CMD=(
        uv run python "${TRAIN_SCRIPT}"
        --detector-name "${name}"
        --segment-mode "${seg_mode}"
        --segment-length "${seg_len}"
        --data-dir "${DATA_DIR}"
        --checkpoint-dir "${detector_ckpt_dir}"
        --log-dir "${detector_log_dir}"
        --batch-size "${BATCH_SIZE}"
        --epochs "${EPOCHS}"
        --learning-rate "${LEARNING_RATE}"
    )

    # スキップサンプリングモードの場合のみスパン長とスキップ率を追加する
    if [[ "${seg_mode}" == "skip" ]]; then
        CMD+=(--span-length "${span_len}" --skip-rate "${skip_rate}")
    fi

    info "実行コマンド: ${CMD[*]}"
    info ""

    # 訓練の実行（ログをファイルに保存しつつ標準出力にも表示する）
    set +e
    "${CMD[@]}" 2>&1 | tee "${detector_log_dir}/train.log"
    exit_code=${PIPESTATUS[0]}
    set -e

    # 訓練結果の確認
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATIONS+=("${DURATION}")

    if [[ ${exit_code} -eq 0 ]]; then
        info "${name} の訓練が完了しました（所要時間: $(format_duration ${DURATION})）"
        RESULTS+=("OK  ${name}: 成功 ($(format_duration ${DURATION}))")
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        warn "${name} の訓練が失敗しました（終了コード: ${exit_code}、所要時間: $(format_duration ${DURATION})）"
        RESULTS+=("NG  ${name}: 失敗 (終了コード: ${exit_code}, $(format_duration ${DURATION}))")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    info ""
done

# ---------------------------------------------------------------------------
# 全体の結果サマリー
# ---------------------------------------------------------------------------
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

info "=========================================="
info "全検出器の訓練結果サマリー"
info "=========================================="
info ""
for result in "${RESULTS[@]}"; do
    info "  ${result}"
done
info ""
info "成功: ${SUCCESS_COUNT}/${DETECTOR_COUNT}  失敗: ${FAIL_COUNT}/${DETECTOR_COUNT}"
info "合計所要時間: $(format_duration ${TOTAL_DURATION})"
info ""
info "チェックポイント: ${CHECKPOINT_DIR}/"
info ""

# 失敗した検出器がある場合は警告メッセージを表示して終了コード1で終了する
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    warn "${FAIL_COUNT}個の検出器の訓練が失敗しました。ログを確認してください。"
    exit 1
fi

info "全ての検出器の訓練が正常に完了しました。"
