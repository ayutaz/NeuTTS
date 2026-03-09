"""設定ファイルの読み込みと管理ユーティリティ。

configs/ ディレクトリのYAMLファイルを読み込み、
訓練・推論の設定を提供する。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# プロジェクトルートディレクトリ (mspoof_tts/ の親)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_TRAIN_CONFIG = _PROJECT_ROOT / "configs" / "detector_train.yaml"
_DEFAULT_INFERENCE_CONFIG = _PROJECT_ROOT / "configs" / "inference.yaml"


@dataclass
class ModelConfig:
    """モデルアーキテクチャの設定。"""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    vocab_size: int = 65536


@dataclass
class OptimizerConfig:
    """オプティマイザの設定。"""

    type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class DetectorConfig:
    """個別検出器の設定。"""

    name: str = ""
    mode: str = "contiguous"  # "contiguous" or "skip"
    segment_length: int = 10
    span: int = 0
    skip_rate: int = 0


@dataclass
class TrainingConfig:
    """訓練設定。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    batch_size: int = 64
    num_epochs: int = 100
    num_workers: int = 4
    seed: int = 42
    gpu: int = 0
    save_dir: str = "checkpoints/detectors"
    save_every: int = 10
    detectors: dict[str, DetectorConfig] = field(default_factory=dict)


@dataclass
class EASConfig:
    """Entropy-Aware Sampling の設定。"""

    top_k: int = 50
    top_p: float = 0.8
    temperature: float = 1.0
    cluster_size: int = 3
    memory_window: int = 15
    alpha: float = 0.2
    beta: float = 0.7
    gamma: float = 0.8


@dataclass
class HierarchicalConfig:
    """階層的デコーディングの設定。"""

    warmup_length: int = 20
    stage_lengths: list[int] = field(default_factory=lambda: [10, 25, 50])
    beam_sizes: list[int] = field(default_factory=lambda: [8, 5, 3])
    rank_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class InferenceConfig:
    """推論設定。"""

    eas: EASConfig = field(default_factory=EASConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)


def _parse_detector_entry(name: str, raw: dict) -> DetectorConfig:
    """YAML辞書から DetectorConfig を組み立てる。"""
    mode = raw.get("mode", "contiguous")
    segment_length = raw.get("segment_length", 0)
    span = raw.get("span", 0)
    skip_rate = raw.get("skip_rate", 0)

    # contiguous モードではセグメント長をそのまま使い、
    # skip モードではスパンからスキップ率で実効長を計算する
    if mode == "skip" and segment_length == 0 and span > 0 and skip_rate > 0:
        segment_length = span // skip_rate

    return DetectorConfig(
        name=name,
        mode=mode,
        segment_length=segment_length,
        span=span,
        skip_rate=skip_rate,
    )


def load_training_config(path: Optional[Path] = None) -> TrainingConfig:
    """訓練設定をYAMLファイルから読み込む。

    Args:
        path: YAMLファイルのパス。Noneの場合はデフォルトパスを使用。

    Returns:
        パースされた TrainingConfig オブジェクト。
    """
    config_path = Path(path) if path is not None else _DEFAULT_TRAIN_CONFIG

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # モデル設定
    model_raw = raw.get("model", {})
    model = ModelConfig(
        d_model=model_raw.get("d_model", 256),
        n_heads=model_raw.get("n_heads", 8),
        n_layers=model_raw.get("n_layers", 4),
        ffn_dim=model_raw.get("ffn_dim", 1024),
        dropout=model_raw.get("dropout", 0.1),
    )

    # オプティマイザ設定
    opt_raw = raw.get("optimizer", {})
    optimizer = OptimizerConfig(
        type=opt_raw.get("type", "AdamW"),
        lr=opt_raw.get("lr", 1e-4),
        weight_decay=opt_raw.get("weight_decay", 1e-4),
    )

    # 訓練パラメータ
    train_raw = raw.get("training", {})
    ckpt_raw = raw.get("checkpoint", {})

    # 検出器設定
    detectors_raw = raw.get("detectors", {})
    detectors = {
        name: _parse_detector_entry(name, cfg)
        for name, cfg in detectors_raw.items()
    }

    return TrainingConfig(
        model=model,
        optimizer=optimizer,
        batch_size=train_raw.get("batch_size", 64),
        num_epochs=train_raw.get("num_epochs", 100),
        num_workers=train_raw.get("num_workers", 4),
        seed=train_raw.get("seed", 42),
        gpu=train_raw.get("gpu", 0),
        save_dir=ckpt_raw.get("save_dir", "checkpoints/detectors"),
        save_every=ckpt_raw.get("save_every", 10),
        detectors=detectors,
    )


def load_inference_config(path: Optional[Path] = None) -> InferenceConfig:
    """推論設定をYAMLファイルから読み込む。

    Args:
        path: YAMLファイルのパス。Noneの場合はデフォルトパスを使用。

    Returns:
        パースされた InferenceConfig オブジェクト。
    """
    config_path = Path(path) if path is not None else _DEFAULT_INFERENCE_CONFIG

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # EAS設定
    eas_raw = raw.get("eas", {})
    eas = EASConfig(
        top_k=eas_raw.get("top_k", 50),
        top_p=eas_raw.get("top_p", 0.8),
        temperature=eas_raw.get("temperature", 1.0),
        cluster_size=eas_raw.get("cluster_size", 3),
        memory_window=eas_raw.get("memory_window", 15),
        alpha=eas_raw.get("alpha", 0.2),
        beta=eas_raw.get("beta", 0.7),
        gamma=eas_raw.get("gamma", 0.8),
    )

    # 階層的デコーディング設定
    hier_raw = raw.get("hierarchical", {})
    hierarchical = HierarchicalConfig(
        warmup_length=hier_raw.get("warmup", 20),
        stage_lengths=hier_raw.get("stage_lengths", [10, 25, 50]),
        beam_sizes=hier_raw.get("beam_sizes", [8, 5, 3]),
        rank_weights=hier_raw.get("rank_weights", [1.0, 1.0, 1.0]),
    )

    return InferenceConfig(eas=eas, hierarchical=hierarchical)


def get_detector_configs() -> dict[str, DetectorConfig]:
    """5つの検出器の設定を返す。

    YAMLファイルを読み込まず、論文で定義された
    ハードコードされた検出器設定を返す便利関数。

    Returns:
        検出器名をキー、DetectorConfig を値とする辞書。
    """
    return {
        "M_10": DetectorConfig(
            name="M_10",
            mode="contiguous",
            segment_length=10,
            span=0,
            skip_rate=0,
        ),
        "M_25": DetectorConfig(
            name="M_25",
            mode="contiguous",
            segment_length=25,
            span=0,
            skip_rate=0,
        ),
        "M_50": DetectorConfig(
            name="M_50",
            mode="contiguous",
            segment_length=50,
            span=0,
            skip_rate=0,
        ),
        "M_50_25": DetectorConfig(
            name="M_50_25",
            mode="skip",
            segment_length=25,
            span=50,
            skip_rate=2,
        ),
        "M_50_10": DetectorConfig(
            name="M_50_10",
            mode="skip",
            segment_length=10,
            span=50,
            skip_rate=5,
        ),
    }
