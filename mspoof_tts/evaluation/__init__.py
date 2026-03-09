"""評価モジュール。

WER、SIM、NISQA、MOSNETなどの評価指標の計算と評価パイプラインを含む。
"""

from mspoof_tts.evaluation.metrics import compute_wer, compute_similarity
from mspoof_tts.evaluation.evaluate import EvaluationPipeline

__all__ = ["compute_wer", "compute_similarity", "EvaluationPipeline"]
