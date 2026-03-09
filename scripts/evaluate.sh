#!/usr/bin/env bash
# =============================================================================
# evaluate.sh
# MSpoof-TTS の合成音声を定量的に評価するスクリプト
#
# 以下の4つの指標を計算する:
#   - WER (Word Error Rate): Whisper-large-v3 による音声認識精度
#   - SIM (Speaker Similarity): WavLM-base-plus-sv による話者類似度
#   - NISQA: 知覚品質スコア
#   - MOSNET: 平均オピニオンスコア推定
#
# 評価データセット: LibriSpeech, LibriTTS, TwistList
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 設定変数（環境変数で上書き可能）
# ---------------------------------------------------------------------------

# 使用する GPU の ID
GPU_ID="${GPU_ID:-0}"

# 合成音声のディレクトリ（inference.sh の出力先）
SYNTH_DIR="${SYNTH_DIR:-./output}"

# リファレンス音声のディレクトリ（話者類似度計算に使用）
REFERENCE_DIR="${REFERENCE_DIR:-./data/reference}"

# 評価結果の保存先
RESULTS_DIR="${RESULTS_DIR:-./results}"

# 評価対象のデータセット名
EVAL_DATASET="${EVAL_DATASET:-libritts}"

# 計算する評価指標（カンマ区切りで指定）
METRICS="${METRICS:-wer,sim,nisqa,mosnet}"

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
  ${0} [オプション]

オプション:
  --synth-dir, -s     合成音声ディレクトリ（デフォルト: ./output）
  --reference-dir, -r リファレンス音声ディレクトリ（デフォルト: ./data/reference）
  --results-dir       評価結果の保存先（デフォルト: ./results）
  --dataset, -d       評価データセット名（デフォルト: libritts）
                      選択肢: librispeech, libritts, twistlist
  --metrics, -m       評価指標（カンマ区切り、デフォルト: wer,sim,nisqa,mosnet）
  --gpu               使用するGPU ID（デフォルト: 0）
  --help, -h          このヘルプメッセージを表示
USAGE
    exit 0
}

# ---------------------------------------------------------------------------
# コマンドライン引数の解析
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synth-dir|-s)
            SYNTH_DIR="$2"; shift 2 ;;
        --reference-dir|-r)
            REFERENCE_DIR="$2"; shift 2 ;;
        --results-dir)
            RESULTS_DIR="$2"; shift 2 ;;
        --dataset|-d)
            EVAL_DATASET="$2"; shift 2 ;;
        --metrics|-m)
            METRICS="$2"; shift 2 ;;
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
if [[ ! -d "${SYNTH_DIR}" ]]; then
    error "合成音声ディレクトリが見つかりません: ${SYNTH_DIR}"
fi

# ---------------------------------------------------------------------------
# 評価の実行
# ---------------------------------------------------------------------------

info "=========================================="
info "MSpoof-TTS 評価スクリプト"
info "=========================================="
info ""
info "設定:"
info "  合成音声:       ${SYNTH_DIR}"
info "  リファレンス:   ${REFERENCE_DIR}"
info "  評価結果保存先: ${RESULTS_DIR}"
info "  データセット:   ${EVAL_DATASET}"
info "  評価指標:       ${METRICS}"
info "  GPU ID:         ${GPU_ID}"
info ""

# 結果ディレクトリの作成
mkdir -p "${RESULTS_DIR}"

# 評価開始時刻を記録
START_TIME=$(date +%s)

# 各評価指標を順番に計算
# IFS（Internal Field Separator）をカンマに設定して指標リストを分割
IFS=',' read -ra METRIC_LIST <<< "${METRICS}"

for metric in "${METRIC_LIST[@]}"; do
    # 前後の空白を除去
    metric=$(echo "${metric}" | tr -d ' ')

    case "${metric}" in
        wer)
            # ---------------------------------------------------------------
            # WER (Word Error Rate)
            # Whisper-large-v3 で合成音声を書き起こし、原文との WER を計算
            # ---------------------------------------------------------------
            info "WER を計算中（Whisper-large-v3 使用）..."
            CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python -m mspoof_tts.evaluation.evaluate \
                --metric wer \
                --synth-dir "${SYNTH_DIR}" \
                --results-dir "${RESULTS_DIR}" \
                --dataset "${EVAL_DATASET}"
            info "WER 計算完了"
            ;;

        sim)
            # ---------------------------------------------------------------
            # SIM (Speaker Similarity)
            # WavLM-base-plus-sv でリファレンスと合成音声の話者埋め込みを比較
            # ---------------------------------------------------------------
            info "話者類似度 (SIM) を計算中（WavLM-base-plus-sv 使用）..."
            CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python -m mspoof_tts.evaluation.evaluate \
                --metric sim \
                --synth-dir "${SYNTH_DIR}" \
                --reference-dir "${REFERENCE_DIR}" \
                --results-dir "${RESULTS_DIR}" \
                --dataset "${EVAL_DATASET}"
            info "SIM 計算完了"
            ;;

        nisqa)
            # ---------------------------------------------------------------
            # NISQA (Non-Intrusive Speech Quality Assessment)
            # NISQA モデルで知覚品質スコアを推定
            # ---------------------------------------------------------------
            info "NISQA スコアを計算中..."
            CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python -m mspoof_tts.evaluation.evaluate \
                --metric nisqa \
                --synth-dir "${SYNTH_DIR}" \
                --results-dir "${RESULTS_DIR}" \
                --dataset "${EVAL_DATASET}"
            info "NISQA 計算完了"
            ;;

        mosnet)
            # ---------------------------------------------------------------
            # MOSNET (Mean Opinion Score Network)
            # MOSNet モデルで平均オピニオンスコアを推定
            # ---------------------------------------------------------------
            info "MOSNET スコアを計算中..."
            CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python -m mspoof_tts.evaluation.evaluate \
                --metric mosnet \
                --synth-dir "${SYNTH_DIR}" \
                --results-dir "${RESULTS_DIR}" \
                --dataset "${EVAL_DATASET}"
            info "MOSNET 計算完了"
            ;;

        *)
            info "不明な評価指標: ${metric}（スキップ）"
            ;;
    esac

    info ""
done

# ---------------------------------------------------------------------------
# 結果サマリーの表示
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

info "=========================================="
info "評価完了"
info "=========================================="
info "所要時間: $((DURATION / 60))分$((DURATION % 60))秒"
info "結果の保存先: ${RESULTS_DIR}/"
info ""

# 結果ファイルが存在する場合に内容を表示
RESULTS_FILE="${RESULTS_DIR}/${EVAL_DATASET}_results.json"
if [[ -f "${RESULTS_FILE}" ]]; then
    info "評価結果:"
    cat "${RESULTS_FILE}"
    info ""
fi

info "論文の Table 2, Table 3 と比較してください。"
info ""
