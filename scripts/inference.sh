#!/usr/bin/env bash
# =============================================================================
# inference.sh
# MSpoof-TTS の推論を実行するスクリプト
#
# 階層的スプーフ誘導デコーディングを用いて、高品質な音声合成を行う。
# ベースTTS（NeuTTS）は凍結状態のまま、訓練済みスプーフ検出器を利用して
# デコーディング過程でトークン列の品質を評価・制御する。
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 設定変数（環境変数で上書き可能）
# ---------------------------------------------------------------------------

# 使用する GPU の ID
GPU_ID="${GPU_ID:-0}"

# チェックポイントディレクトリ（訓練済み検出器の保存先）
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# 入力テキストファイル（1行1文）
INPUT_TEXT="${INPUT_TEXT:-}"

# リファレンス音声ファイル（話者特性の条件付けに使用）
REFERENCE_AUDIO="${REFERENCE_AUDIO:-}"

# 出力ディレクトリ
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# ---------------------------------------------------------------------------
# デコーディングハイパーパラメータ
# ---------------------------------------------------------------------------

# Entropy-Aware Sampling (EAS) のパラメータ
TOP_K="${TOP_K:-50}"                     # top-k サンプリングのk値
TOP_P="${TOP_P:-0.8}"                    # nucleus sampling の累積確率閾値
TEMPERATURE="${TEMPERATURE:-1.0}"        # サンプリング温度
EAS_ALPHA="${EAS_ALPHA:-0.2}"            # 繰り返しペナルティの強度
EAS_BETA="${EAS_BETA:-0.7}"              # 時間減衰率
EAS_GAMMA="${EAS_GAMMA:-0.8}"            # Nucleus sampling の閾値
CLUSTER_SIZE="${CLUSTER_SIZE:-3}"        # エントロピー閾値パラメータ k_e
MEMORY_WINDOW="${MEMORY_WINDOW:-15}"     # メモリバッファのウィンドウサイズ W

# 階層的ビームサーチのパラメータ
WARMUP_LENGTH="${WARMUP_LENGTH:-20}"     # ウォームアップフェーズの長さ L_w
BEAM_SIZE_0="${BEAM_SIZE_0:-8}"          # Stage 1 のビーム幅 B0
BEAM_SIZE_1="${BEAM_SIZE_1:-5}"          # Stage 2 のビーム幅 B1
BEAM_SIZE_2="${BEAM_SIZE_2:-3}"          # Stage 3 のビーム幅 B2
STAGE_LENGTH_1="${STAGE_LENGTH_1:-10}"   # Stage 1 のセグメント長 L1
STAGE_LENGTH_2="${STAGE_LENGTH_2:-25}"   # Stage 2 のセグメント長 L2
STAGE_LENGTH_3="${STAGE_LENGTH_3:-50}"   # Stage 3 のセグメント長 L3

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

# 使い方を表示する関数
usage() {
    cat <<USAGE
使い方:
  ${0} --input <テキストファイル> --reference <リファレンス音声> [オプション]

必須引数:
  --input, -i       入力テキストファイルのパス（1行1文）
  --reference, -r   リファレンス音声ファイルのパス

オプション:
  --output, -o      出力ディレクトリ（デフォルト: ./output）
  --checkpoint, -c  チェックポイントディレクトリ（デフォルト: ./checkpoints）
  --gpu             使用するGPU ID（デフォルト: 0）
  --help, -h        このヘルプメッセージを表示

環境変数でデコーディングパラメータを制御できます:
  TOP_K, TOP_P, TEMPERATURE, EAS_ALPHA, EAS_BETA, EAS_GAMMA,
  CLUSTER_SIZE, MEMORY_WINDOW, WARMUP_LENGTH,
  BEAM_SIZE_0, BEAM_SIZE_1, BEAM_SIZE_2,
  STAGE_LENGTH_1, STAGE_LENGTH_2, STAGE_LENGTH_3
USAGE
    exit 0
}

# ---------------------------------------------------------------------------
# コマンドライン引数の解析
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)
            INPUT_TEXT="$2"; shift 2 ;;
        --reference|-r)
            REFERENCE_AUDIO="$2"; shift 2 ;;
        --output|-o)
            OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint|-c)
            CHECKPOINT_DIR="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --help|-h)
            usage ;;
        *)
            error "不明な引数: $1（--help で使い方を確認してください）" ;;
    esac
done

# ---------------------------------------------------------------------------
# 入力の検証
# ---------------------------------------------------------------------------
if [[ -z "${INPUT_TEXT}" ]]; then
    error "入力テキストファイルを指定してください（--input）"
fi

if [[ -z "${REFERENCE_AUDIO}" ]]; then
    error "リファレンス音声ファイルを指定してください（--reference）"
fi

if [[ ! -f "${INPUT_TEXT}" ]]; then
    error "入力テキストファイルが見つかりません: ${INPUT_TEXT}"
fi

if [[ ! -f "${REFERENCE_AUDIO}" ]]; then
    error "リファレンス音声ファイルが見つかりません: ${REFERENCE_AUDIO}"
fi

# 5つの検出器チェックポイントの存在確認
DETECTORS=("M_50" "M_25" "M_10" "M_50_25" "M_50_10")
for det in "${DETECTORS[@]}"; do
    if [[ ! -d "${CHECKPOINT_DIR}/${det}" ]]; then
        error "検出器のチェックポイントが見つかりません: ${CHECKPOINT_DIR}/${det}"
    fi
done

# ---------------------------------------------------------------------------
# 推論の実行
# ---------------------------------------------------------------------------

info "=========================================="
info "MSpoof-TTS 推論"
info "=========================================="
info ""
info "入力テキスト:     ${INPUT_TEXT}"
info "リファレンス音声: ${REFERENCE_AUDIO}"
info "出力先:           ${OUTPUT_DIR}"
info "チェックポイント: ${CHECKPOINT_DIR}"
info "GPU ID:           ${GPU_ID}"
info ""
info "EAS パラメータ:"
info "  top-k=${TOP_K}, top-p=${TOP_P}, temperature=${TEMPERATURE}"
info "  alpha=${EAS_ALPHA}, beta=${EAS_BETA}, gamma=${EAS_GAMMA}"
info "  cluster_size=${CLUSTER_SIZE}, memory_window=${MEMORY_WINDOW}"
info ""
info "階層的デコーディング パラメータ:"
info "  warmup=${WARMUP_LENGTH}"
info "  beam_sizes=(${BEAM_SIZE_0}, ${BEAM_SIZE_1}, ${BEAM_SIZE_2})"
info "  stage_lengths=(${STAGE_LENGTH_1}, ${STAGE_LENGTH_2}, ${STAGE_LENGTH_3})"
info ""

# 出力ディレクトリの作成
mkdir -p "${OUTPUT_DIR}"

# 推論スクリプトの実行
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 inference.py \
    --input-text "${INPUT_TEXT}" \
    --reference-audio "${REFERENCE_AUDIO}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --temperature "${TEMPERATURE}" \
    --eas-alpha "${EAS_ALPHA}" \
    --eas-beta "${EAS_BETA}" \
    --eas-gamma "${EAS_GAMMA}" \
    --cluster-size "${CLUSTER_SIZE}" \
    --memory-window "${MEMORY_WINDOW}" \
    --warmup-length "${WARMUP_LENGTH}" \
    --beam-sizes "${BEAM_SIZE_0}" "${BEAM_SIZE_1}" "${BEAM_SIZE_2}" \
    --stage-lengths "${STAGE_LENGTH_1}" "${STAGE_LENGTH_2}" "${STAGE_LENGTH_3}"

info ""
info "推論完了"
info "出力ファイル: ${OUTPUT_DIR}/"
info ""
info "次のステップ:"
info "  scripts/evaluate.sh で合成音声の品質を評価する"
info ""
