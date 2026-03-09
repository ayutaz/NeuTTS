#!/usr/bin/env bash
# =============================================================================
# inference.sh
# MSpoof-TTS の推論を実行するスクリプト
#
# EAS（Entropy-Aware Sampling）モードと階層的スプーフ誘導デコーディングモードの
# 両方をサポートする。ベースTTS（NeuTTS）は凍結状態のまま、訓練済みスプーフ
# 検出器を利用してデコーディング過程でトークン列の品質を評価・制御する。
#
# 使い方:
#   # EASモード（デフォルト）
#   ./scripts/inference.sh --input "Hello world" --reference samples/dave.wav \
#       --output output.wav --mode eas
#
#   # 階層的デコーディングモード
#   ./scripts/inference.sh --input "Hello world" --reference samples/dave.wav \
#       --output output.wav --mode hierarchical \
#       --checkpoint-dir checkpoints/detectors
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト値（環境変数で上書き可能）
# ---------------------------------------------------------------------------

# 使用する GPU の ID
GPU_ID="${GPU_ID:-0}"

# デコーディングモード: eas または hierarchical
MODE="${MODE:-eas}"

# チェックポイントディレクトリ（階層的モード時に必須）
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# 入力テキスト（直接指定）
INPUT_TEXT="${INPUT_TEXT:-}"

# 入力テキストファイル（--input-file で渡す場合）
INPUT_FILE="${INPUT_FILE:-}"

# リファレンス音声ファイル（話者特性の条件付けに使用）
REFERENCE_AUDIO="${REFERENCE_AUDIO:-}"

# リファレンスのテキスト（未指定ならリファレンス音声と同名の .txt を自動検出）
REFERENCE_TEXT="${REFERENCE_TEXT:-}"

# 出力ファイルパス
OUTPUT="${OUTPUT:-./output.wav}"

# 推論設定YAMLファイル
CONFIG="${CONFIG:-}"

# ---------------------------------------------------------------------------
# Entropy-Aware Sampling (EAS) のハイパーパラメータ
# ---------------------------------------------------------------------------

TOP_K="${TOP_K:-50}"                     # top-k サンプリングのk値
TOP_P="${TOP_P:-0.8}"                    # nucleus sampling の累積確率閾値
TEMPERATURE="${TEMPERATURE:-1.0}"        # サンプリング温度
EAS_ALPHA="${EAS_ALPHA:-0.2}"            # 繰り返しペナルティの強度
EAS_BETA="${EAS_BETA:-0.7}"              # 時間減衰率
EAS_GAMMA="${EAS_GAMMA:-0.8}"            # クリッピング上限
CLUSTER_SIZE="${CLUSTER_SIZE:-3}"        # エントロピー閾値パラメータ k_e
MEMORY_WINDOW="${MEMORY_WINDOW:-15}"     # メモリバッファのウィンドウサイズ W

# ---------------------------------------------------------------------------
# 階層的ビームサーチのハイパーパラメータ
# ---------------------------------------------------------------------------

WARMUP="${WARMUP:-20}"                          # ウォームアップフェーズの長さ L_w
BEAM_SIZES="${BEAM_SIZES:-8,5,3}"               # 各ステージのビーム幅 B0,B1,B2（カンマ区切り）
STAGE_LENGTHS="${STAGE_LENGTHS:-10,25,50}"       # 各ステージのセグメント長 L1,L2,L3（カンマ区切り）
RANK_WEIGHTS="${RANK_WEIGHTS:-1.0,1.0,1.0}"     # ランク集約の重み w_50,w_25,w_10（カンマ区切り）

# ---------------------------------------------------------------------------
# その他の設定
# ---------------------------------------------------------------------------

MAX_TOKENS="${MAX_TOKENS:-2048}"                # 最大生成トークン数
SEED="${SEED:-42}"                              # 乱数シード
DEVICE="${DEVICE:-}"                            # 推論デバイス（空なら自動検出）
BACKBONE_REPO="${BACKBONE_REPO:-}"              # NeuTTS backbone のリポジトリ名
CODEC_REPO="${CODEC_REPO:-}"                    # NeuCodec のリポジトリ名
USE_FP16="${USE_FP16:-false}"                   # 半精度推論を使うかどうか

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
  ${0} --input <テキスト> --reference <リファレンス音声> [オプション]

必須引数:
  --input, -i           合成するテキスト（直接指定）
  --input-file          入力テキストファイルのパス（1行1文、--input と排他）
  --reference, -r       リファレンス音声ファイルのパス

デコーディングモード:
  --mode, -m            デコーディングモード: eas または hierarchical（デフォルト: eas）

出力・設定:
  --output, -o          出力WAVファイルのパス（デフォルト: ./output.wav）
  --config              推論設定YAMLファイルのパス（configs/inference.yaml）
  --checkpoint-dir, -c  検出器チェックポイントのディレクトリ（hierarchicalモード時に必須）
  --gpu                 使用するGPU ID（デフォルト: 0）
  --device              推論デバイス: cpu / cuda / mps（デフォルト: 自動検出）
  --seed                乱数シード（デフォルト: 42）
  --max-tokens          最大生成トークン数（デフォルト: 2048）
  --backbone-repo       NeuTTS backbone のリポジトリ名またはパス
  --codec-repo          NeuCodec のリポジトリ名またはパス
  --fp16                半精度（FP16）で推論を行う

EAS パラメータ（CLIまたは環境変数で指定可能）:
  --top-k               top-k サンプリングのk値（デフォルト: 50 / 環境変数: TOP_K）
  --top-p               nucleus sampling の閾値（デフォルト: 0.8 / 環境変数: TOP_P）
  --temperature         サンプリング温度（デフォルト: 1.0 / 環境変数: TEMPERATURE）
  --eas-alpha           繰り返しペナルティ強度（デフォルト: 0.2 / 環境変数: EAS_ALPHA）
  --eas-beta            時間減衰率（デフォルト: 0.7 / 環境変数: EAS_BETA）
  --eas-gamma           クリッピング上限（デフォルト: 0.8 / 環境変数: EAS_GAMMA）
  --cluster-size        エントロピー閾値 k_e（デフォルト: 3 / 環境変数: CLUSTER_SIZE）
  --memory-window       メモリウィンドウ W（デフォルト: 15 / 環境変数: MEMORY_WINDOW）

階層的デコーディングのパラメータ（CLIまたは環境変数で指定可能）:
  --warmup              ウォームアップ長 L_w（デフォルト: 20 / 環境変数: WARMUP）
  --beam-sizes          ビーム幅 B0,B1,B2（カンマ区切り、デフォルト: 8,5,3 / 環境変数: BEAM_SIZES）
  --stage-lengths       セグメント長 L1,L2,L3（カンマ区切り、デフォルト: 10,25,50 / 環境変数: STAGE_LENGTHS）
  --rank-weights        ランク重み w_50,w_25,w_10（カンマ区切り、デフォルト: 1.0,1.0,1.0 / 環境変数: RANK_WEIGHTS）

その他:
  --reference-text      リファレンス音声のテキスト（未指定なら .txt から自動検出）
  --help, -h            このヘルプメッセージを表示

使用例:
  # EASモード（デフォルト）
  ${0} --input "Hello world" --reference samples/dave.wav --output output.wav

  # EASモード + パラメータ調整
  TOP_K=30 TEMPERATURE=0.9 ${0} --input "Hello world" --reference samples/dave.wav

  # 階層的デコーディングモード
  ${0} --input "Hello world" --reference samples/dave.wav --output output.wav \\
      --mode hierarchical --checkpoint-dir checkpoints/detectors

  # 設定YAMLを指定
  ${0} --input "Hello world" --reference samples/dave.wav --config configs/inference.yaml
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
        --input-file)
            INPUT_FILE="$2"; shift 2 ;;
        --reference|-r)
            REFERENCE_AUDIO="$2"; shift 2 ;;
        --reference-text)
            REFERENCE_TEXT="$2"; shift 2 ;;
        --output|-o)
            OUTPUT="$2"; shift 2 ;;
        --mode|-m)
            MODE="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --checkpoint-dir|-c)
            CHECKPOINT_DIR="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --max-tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        --backbone-repo)
            BACKBONE_REPO="$2"; shift 2 ;;
        --codec-repo)
            CODEC_REPO="$2"; shift 2 ;;
        --fp16)
            USE_FP16="true"; shift ;;
        # EAS パラメータ
        --top-k)
            TOP_K="$2"; shift 2 ;;
        --top-p)
            TOP_P="$2"; shift 2 ;;
        --temperature)
            TEMPERATURE="$2"; shift 2 ;;
        --eas-alpha)
            EAS_ALPHA="$2"; shift 2 ;;
        --eas-beta)
            EAS_BETA="$2"; shift 2 ;;
        --eas-gamma)
            EAS_GAMMA="$2"; shift 2 ;;
        --cluster-size)
            CLUSTER_SIZE="$2"; shift 2 ;;
        --memory-window)
            MEMORY_WINDOW="$2"; shift 2 ;;
        # 階層的デコーディングのパラメータ
        --warmup)
            WARMUP="$2"; shift 2 ;;
        --beam-sizes)
            BEAM_SIZES="$2"; shift 2 ;;
        --stage-lengths)
            STAGE_LENGTHS="$2"; shift 2 ;;
        --rank-weights)
            RANK_WEIGHTS="$2"; shift 2 ;;
        --help|-h)
            usage ;;
        *)
            error "不明な引数: $1（--help で使い方を確認してください）" ;;
    esac
done

# ---------------------------------------------------------------------------
# 入力の検証
# ---------------------------------------------------------------------------

# モードの検証
if [[ "${MODE}" != "eas" && "${MODE}" != "hierarchical" ]]; then
    error "不正なモードです: ${MODE}（eas または hierarchical を指定してください）"
fi

# 入力テキストの検証（--input か --input-file のいずれかが必要）
if [[ -z "${INPUT_TEXT}" && -z "${INPUT_FILE}" ]]; then
    error "入力テキストを指定してください（--input または --input-file）"
fi

if [[ -n "${INPUT_TEXT}" && -n "${INPUT_FILE}" ]]; then
    error "--input と --input-file は同時に指定できません"
fi

# 入力ファイルの存在確認
if [[ -n "${INPUT_FILE}" && ! -f "${INPUT_FILE}" ]]; then
    error "入力テキストファイルが見つかりません: ${INPUT_FILE}"
fi

# リファレンス音声の検証
if [[ -z "${REFERENCE_AUDIO}" ]]; then
    error "リファレンス音声ファイルを指定してください（--reference）"
fi

if [[ ! -f "${REFERENCE_AUDIO}" ]]; then
    error "リファレンス音声ファイルが見つかりません: ${REFERENCE_AUDIO}"
fi

# 階層的モードではチェックポイントディレクトリが必須
if [[ "${MODE}" == "hierarchical" ]]; then
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        error "hierarchical モードでは --checkpoint-dir の指定が必要です"
    fi
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        error "チェックポイントディレクトリが見つかりません: ${CHECKPOINT_DIR}"
    fi
fi

# 設定YAMLファイルの存在確認（指定されている場合）
if [[ -n "${CONFIG}" && ! -f "${CONFIG}" ]]; then
    error "設定ファイルが見つかりません: ${CONFIG}"
fi

# ---------------------------------------------------------------------------
# カンマ区切りの値をスペース区切りに変換するヘルパー関数
# 例: "8,5,3" → "8 5 3"
# ---------------------------------------------------------------------------
comma_to_spaces() {
    echo "$1" | tr ',' ' '
}

# ---------------------------------------------------------------------------
# 推論コマンドの構築
# ---------------------------------------------------------------------------

# 基本引数の組み立て
CMD_ARGS=()

# 入力テキスト（直接指定 or ファイル）
if [[ -n "${INPUT_TEXT}" ]]; then
    CMD_ARGS+=(--input-text "${INPUT_TEXT}")
else
    CMD_ARGS+=(--input-file "${INPUT_FILE}")
fi

CMD_ARGS+=(--reference-audio "${REFERENCE_AUDIO}")
CMD_ARGS+=(--output "${OUTPUT}")
CMD_ARGS+=(--mode "${MODE}")
CMD_ARGS+=(--seed "${SEED}")
CMD_ARGS+=(--max-tokens "${MAX_TOKENS}")

# 設定YAMLファイル（指定されている場合）
if [[ -n "${CONFIG}" ]]; then
    CMD_ARGS+=(--config "${CONFIG}")
fi

# チェックポイントディレクトリ（階層的モードまたは明示的に指定されている場合）
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    CMD_ARGS+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

# リファレンステキスト（指定されている場合）
if [[ -n "${REFERENCE_TEXT}" ]]; then
    CMD_ARGS+=(--reference-text "${REFERENCE_TEXT}")
fi

# デバイス設定
if [[ -n "${DEVICE}" ]]; then
    CMD_ARGS+=(--device "${DEVICE}")
fi

# backbone / codec リポジトリ
if [[ -n "${BACKBONE_REPO}" ]]; then
    CMD_ARGS+=(--backbone-repo "${BACKBONE_REPO}")
fi
if [[ -n "${CODEC_REPO}" ]]; then
    CMD_ARGS+=(--codec-repo "${CODEC_REPO}")
fi

# FP16
if [[ "${USE_FP16}" == "true" ]]; then
    CMD_ARGS+=(--use-fp16)
fi

# EAS パラメータ
CMD_ARGS+=(--top-k "${TOP_K}")
CMD_ARGS+=(--top-p "${TOP_P}")
CMD_ARGS+=(--temperature "${TEMPERATURE}")
CMD_ARGS+=(--eas-alpha "${EAS_ALPHA}")
CMD_ARGS+=(--eas-beta "${EAS_BETA}")
CMD_ARGS+=(--eas-gamma "${EAS_GAMMA}")
CMD_ARGS+=(--cluster-size "${CLUSTER_SIZE}")
CMD_ARGS+=(--memory-window "${MEMORY_WINDOW}")

# 階層的デコーディングのパラメータ（モードに関係なく常に渡す — Python側で必要に応じて使用）
CMD_ARGS+=(--warmup-length "${WARMUP}")

# カンマ区切りをスペース区切りに変換して配列として渡す
read -r -a BEAM_SIZES_ARR <<< "$(comma_to_spaces "${BEAM_SIZES}")"
CMD_ARGS+=(--beam-sizes "${BEAM_SIZES_ARR[@]}")

read -r -a STAGE_LENGTHS_ARR <<< "$(comma_to_spaces "${STAGE_LENGTHS}")"
CMD_ARGS+=(--stage-lengths "${STAGE_LENGTHS_ARR[@]}")

read -r -a RANK_WEIGHTS_ARR <<< "$(comma_to_spaces "${RANK_WEIGHTS}")"
CMD_ARGS+=(--rank-weights "${RANK_WEIGHTS_ARR[@]}")

# ---------------------------------------------------------------------------
# 設定内容の表示
# ---------------------------------------------------------------------------

info "=========================================="
info "MSpoof-TTS 推論"
info "=========================================="
info ""
info "モード:             ${MODE}"
info "入力テキスト:       ${INPUT_TEXT:-${INPUT_FILE}}"
info "リファレンス音声:   ${REFERENCE_AUDIO}"
info "出力先:             ${OUTPUT}"
info "GPU ID:             ${GPU_ID}"
if [[ -n "${CONFIG}" ]]; then
    info "設定ファイル:       ${CONFIG}"
fi
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    info "チェックポイント:   ${CHECKPOINT_DIR}"
fi
info ""
info "EAS パラメータ:"
info "  top-k=${TOP_K}, top-p=${TOP_P}, temperature=${TEMPERATURE}"
info "  alpha=${EAS_ALPHA}, beta=${EAS_BETA}, gamma=${EAS_GAMMA}"
info "  cluster_size=${CLUSTER_SIZE}, memory_window=${MEMORY_WINDOW}"
info ""

if [[ "${MODE}" == "hierarchical" ]]; then
    info "階層的デコーディング パラメータ:"
    info "  warmup=${WARMUP}"
    info "  beam_sizes=(${BEAM_SIZES})"
    info "  stage_lengths=(${STAGE_LENGTHS})"
    info "  rank_weights=(${RANK_WEIGHTS})"
    info ""
fi

# ---------------------------------------------------------------------------
# 推論の実行
# ---------------------------------------------------------------------------

# 出力ディレクトリの作成
mkdir -p "$(dirname "${OUTPUT}")"

info "推論を開始します..."
info ""

# uv run python を使って inference.py を実行する
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python inference.py "${CMD_ARGS[@]}"

info ""
info "推論完了"
info "出力ファイル: ${OUTPUT}"
info ""
info "次のステップ:"
info "  scripts/evaluate.sh で合成音声の品質を評価する"
info ""
