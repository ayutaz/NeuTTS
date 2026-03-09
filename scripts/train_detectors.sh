#!/usr/bin/env bash
# =============================================================================
# train_detectors.sh
# 5つのマルチ解像度スプーフ検出器を順番に訓練するスクリプト
#
# 各検出器は独立したパラメータを持つ Conformer ベースの二値分類器であり、
# 異なる時間解像度でリアル/フェイクのトークンセグメントを識別する。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 設定変数（環境変数で上書き可能）
# ---------------------------------------------------------------------------

# 使用する GPU の ID（複数GPU環境の場合に変更）
GPU_ID="${GPU_ID:-0}"

# バッチサイズ（GPUメモリに応じて調整）
BATCH_SIZE="${BATCH_SIZE:-64}"

# 訓練エポック数
EPOCHS="${EPOCHS:-100}"

# 学習率
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# Weight decay
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# 訓練データのルートディレクトリ
DATA_DIR="${DATA_DIR:-./data/libritts}"

# チェックポイントの保存先
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# ログの保存先
LOG_DIR="${LOG_DIR:-./logs}"

# train.py のパス
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./train.py}"

# ---------------------------------------------------------------------------
# 5つの検出器の設定
# 各検出器はセグメント構築方法（連続クロッピング or スキップサンプリング）と
# 入力トークン長が異なる
# ---------------------------------------------------------------------------

# 検出器名の配列
DETECTOR_NAMES=(
    "M_50"       # 長さ50の連続部分列で訓練
    "M_25"       # 長さ25の連続部分列で訓練
    "M_10"       # 長さ10の連続部分列で訓練
    "M_50_25"    # 50トークンスパンからr=2でスキップサンプリング（25トークン）
    "M_50_10"    # 50トークンスパンからr=5でスキップサンプリング（10トークン）
)

# セグメント構築方法（contiguous: 連続クロッピング, skip: スキップサンプリング）
SEGMENT_MODES=(
    "contiguous"
    "contiguous"
    "contiguous"
    "skip"
    "skip"
)

# セグメント長（モデルに入力されるトークン数）
SEGMENT_LENGTHS=(
    50   # M_50: 連続50トークン
    25   # M_25: 連続25トークン
    10   # M_10: 連続10トークン
    25   # M_50←25: 50トークンから2つおきに抽出した25トークン
    10   # M_50←10: 50トークンから5つおきに抽出した10トークン
)

# スキップサンプリング時のスパン長（連続クロッピングの場合は0）
SPAN_LENGTHS=(
    0    # M_50: 連続クロッピングなのでスパン不要
    0    # M_25: 同上
    0    # M_10: 同上
    50   # M_50←25: 50トークンスパンからサンプリング
    50   # M_50←10: 50トークンスパンからサンプリング
)

# スキップサンプリング時のスキップ率（連続クロッピングの場合は0）
SKIP_RATES=(
    0    # M_50: 連続クロッピングなのでスキップ不要
    0    # M_25: 同上
    0    # M_10: 同上
    2    # M_50←25: 1つおきにサンプリング（r=2）
    5    # M_50←10: 4つおきにサンプリング（r=5）
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

# 経過時間を人間が読みやすい形式で表示する関数
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d時間%02d分%02d秒" "${hours}" "${minutes}" "${secs}"
}

# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

info "=========================================="
info "マルチ解像度スプーフ検出器 訓練スクリプト"
info "=========================================="
info ""
info "設定:"
info "  GPU ID:        ${GPU_ID}"
info "  バッチサイズ:  ${BATCH_SIZE}"
info "  エポック数:    ${EPOCHS}"
info "  学習率:        ${LEARNING_RATE}"
info "  Weight decay:  ${WEIGHT_DECAY}"
info "  データ:        ${DATA_DIR}"
info "  チェックポイント: ${CHECKPOINT_DIR}"
info "  ログ:          ${LOG_DIR}"
info ""

# train.py の存在確認
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    error "訓練スクリプトが見つかりません: ${TRAIN_SCRIPT}"
fi

# ディレクトリの作成
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"

# 全体の開始時刻を記録
TOTAL_START_TIME=$(date +%s)

# 訓練の成功/失敗を記録する配列
declare -a RESULTS=()

# ---------------------------------------------------------------------------
# 5つの検出器を順番に訓練
# ---------------------------------------------------------------------------
for i in "${!DETECTOR_NAMES[@]}"; do
    name="${DETECTOR_NAMES[$i]}"
    mode="${SEGMENT_MODES[$i]}"
    seg_len="${SEGMENT_LENGTHS[$i]}"
    span_len="${SPAN_LENGTHS[$i]}"
    skip_rate="${SKIP_RATES[$i]}"

    info "=========================================="
    info "検出器 $((i + 1))/5: ${name} の訓練を開始"
    info "  セグメント方式: ${mode}"
    info "  セグメント長:   ${seg_len}"
    if [[ "${mode}" == "skip" ]]; then
        info "  スパン長:       ${span_len}"
        info "  スキップ率:     ${skip_rate}"
    fi
    info "=========================================="

    # 検出器ごとのチェックポイントとログディレクトリ
    detector_ckpt_dir="${CHECKPOINT_DIR}/${name}"
    detector_log_dir="${LOG_DIR}/${name}"
    mkdir -p "${detector_ckpt_dir}"
    mkdir -p "${detector_log_dir}"

    # 訓練開始時刻を記録
    START_TIME=$(date +%s)

    # 訓練コマンドの構築と実行
    # CUDA_VISIBLE_DEVICES で使用GPUを指定
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${TRAIN_SCRIPT}" \
        --detector-name "${name}" \
        --segment-mode "${mode}" \
        --segment-length "${seg_len}" \
        --span-length "${span_len}" \
        --skip-rate "${skip_rate}" \
        --data-dir "${DATA_DIR}" \
        --checkpoint-dir "${detector_ckpt_dir}" \
        --log-dir "${detector_log_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --learning-rate "${LEARNING_RATE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --d-model 256 \
        --num-heads 8 \
        --num-layers 4 \
        --ffn-dim 1024 \
        --dropout 0.1 \
        2>&1 | tee "${detector_log_dir}/train.log"

    # 訓練結果の確認
    exit_code=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [[ ${exit_code} -eq 0 ]]; then
        info "${name} の訓練が完了しました（所要時間: $(format_duration ${DURATION})）"
        RESULTS+=("${name}: 成功 ($(format_duration ${DURATION}))")
    else
        info "${name} の訓練が失敗しました（終了コード: ${exit_code}）"
        RESULTS+=("${name}: 失敗 (終了コード: ${exit_code})")
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
for result in "${RESULTS[@]}"; do
    info "  ${result}"
done
info ""
info "合計所要時間: $(format_duration ${TOTAL_DURATION})"
info ""
info "次のステップ:"
info "  1. 各検出器の訓練ログを確認する: ${LOG_DIR}/"
info "  2. チェックポイントを確認する: ${CHECKPOINT_DIR}/"
info "  3. scripts/inference.sh で推論を実行する"
info ""
