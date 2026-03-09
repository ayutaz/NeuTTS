"""階層的サンプリングと段階的Discriminator枝刈りの実装 — Algorithm 2。

EASの上に構築された階層的スプーフガイド付きデコーディング戦略。
3ステージのビームサーチにより、短スパン -> 中距離 -> 長スパンの順に
候補を枝刈りし、最終的にマルチ解像度ランク集約で最良候補を選出する。

ハイパーパラメータ:
    - warmup (L_w): 20
    - stage_lengths (L1, L2, L3): [10, 25, 50]
    - beam_sizes (B0, B1, B2): [8, 5, 3]
    - rank_weights (w_50, w_25, w_10): [1.0, 1.0, 1.0] (均等)
"""

import copy
import logging
from typing import Optional

import torch

from mspoof_tts.models.multi_resolution import MultiResolutionDetector
from mspoof_tts.sampling.eas import EntropyAwareSampler

logger = logging.getLogger(__name__)


class HierarchicalDecoder:
    """階層的スプーフガイド付きデコーダー。

    ウォームアップフェーズでEASのみで初期トークンを生成した後、
    50トークンごとに3ステージのビームサーチ・枝刈り・ランク集約を
    反復的に実行する。

    Args:
        eas_sampler: EASサンプラーインスタンス。
        detectors: マルチ解像度検出器インスタンス。
        warmup: ウォームアップ長 (L_w)。
        stage_lengths: 各ステージのトークン長 [L1, L2, L3]。
        beam_sizes: 各ステージのビームサイズ [B0, B1, B2]。
        rank_weights: ランク集約の重み [w_50, w_25, w_10]。
        max_length: 最大生成トークン長。
    """

    def __init__(
        self,
        eas_sampler: EntropyAwareSampler,
        detectors: MultiResolutionDetector,
        warmup: int = 20,
        stage_lengths: Optional[list[int]] = None,
        beam_sizes: Optional[list[int]] = None,
        rank_weights: Optional[list[float]] = None,
        max_length: int = 2048,
    ) -> None:
        self.eas_sampler = eas_sampler
        self.detectors = detectors
        self.warmup = warmup
        self.stage_lengths = stage_lengths or [10, 25, 50]
        self.beam_sizes = beam_sizes or [8, 5, 3]
        self.rank_weights = rank_weights or [1.0, 1.0, 1.0]
        self.max_length = max_length
        self.L1, self.L2, self.L3 = self.stage_lengths
        self.B0, self.B1, self.B2 = self.beam_sizes

    # ------------------------------------------------------------------
    # ヘルパー: 指定のEASサンプラーでnトークン生成する
    # ------------------------------------------------------------------

    def _generate_n_tokens(
        self,
        model: object,
        prefix: torch.Tensor,
        n_tokens: int,
        sampler: EntropyAwareSampler,
    ) -> torch.Tensor:
        """指定のEASサンプラーでnトークン生成する。

        Args:
            model: ARバックボーンモデル (HuggingFace CausalLM互換)。
            prefix: 入力トークン列 (seq_len,)。
            n_tokens: 生成するトークン数。
            sampler: 使用するEASサンプラー。

        Returns:
            prefix + 生成トークンを連結したトークン列 (seq_len + n_tokens,)。
        """
        current = prefix.clone()
        for _ in range(n_tokens):
            with torch.no_grad():
                input_ids = current.unsqueeze(0)  # (1, seq_len)
                outputs = model(input_ids)  # type: ignore[operator]
                logits = outputs.logits[0, -1, :]  # (vocab_size,)
            token = sampler.sample_step(logits)
            current = torch.cat([current, token.reshape(1)])
        return current

    # ------------------------------------------------------------------
    # ウォームアップフェーズ
    # ------------------------------------------------------------------

    def _warmup_phase(
        self, model: object, prefix: torch.Tensor
    ) -> torch.Tensor:
        """ウォームアップフェーズ: EASのみでL_wトークンを生成する。

        Discriminatorによる枝刈りを行わず、ARモデルの自然な生成に委ねて
        初期セグメントを安定化させる。

        Args:
            model: ARバックボーンモデル。
            prefix: 入力プレフィックス (seq_len,)。

        Returns:
            ウォームアップ後のトークン列 (seq_len + L_w,)。
        """
        self.eas_sampler.reset()
        logger.info("ウォームアップフェーズ: %d トークン生成中...", self.warmup)
        result = self._generate_n_tokens(model, prefix, self.warmup, self.eas_sampler)
        logger.info("ウォームアップ完了。列長: %d", result.size(0))
        return result

    # ------------------------------------------------------------------
    # Stage 1: 短スパン枝刈り (L1 = 10 トークン)
    # ------------------------------------------------------------------

    def _stage1_prune(
        self,
        model: object,
        prefix: torch.Tensor,
    ) -> list[tuple[torch.Tensor, EntropyAwareSampler]]:
        """Stage 1: 短スパン枝刈り。

        B0個の独立EASサンプラーでL1トークンを生成し、M_10でスコアリング後、
        上位B1個を選択する。

        Args:
            model: ARバックボーンモデル。
            prefix: 現在のトークン列 (seq_len,)。

        Returns:
            Stage 1を通過したB1個の (候補テンソル, サンプラー) タプルのリスト。
        """
        # B0個の独立EASサンプラーをフォーク
        samplers = [copy.deepcopy(self.eas_sampler) for _ in range(self.B0)]

        # 各ビームでL1トークンを生成
        candidates: list[torch.Tensor] = []
        for b in range(self.B0):
            candidate = self._generate_n_tokens(model, prefix, self.L1, samplers[b])
            candidates.append(candidate)

        # M_10でスコアリング: 各候補の末尾L1トークンをバッチ化
        last_tokens = torch.stack([c[-self.L1:] for c in candidates])  # (B0, L1)
        with torch.no_grad():
            scores = self.detectors.score_short(last_tokens)  # (B0,)

        # 上位B1個を選択
        top_indices = torch.argsort(scores, descending=True)[: self.B1]
        logger.info(
            "Stage 1 枝刈り: B0=%d → B1=%d (スコア上位: %s)",
            self.B0,
            self.B1,
            scores[top_indices].tolist(),
        )

        selected: list[tuple[torch.Tensor, EntropyAwareSampler]] = [
            (candidates[idx.item()], samplers[idx.item()])
            for idx in top_indices
        ]
        return selected

    # ------------------------------------------------------------------
    # Stage 2: 中距離枝刈り (L2 = 25 トークン)
    # ------------------------------------------------------------------

    def _stage2_prune(
        self,
        model: object,
        candidates: list[tuple[torch.Tensor, EntropyAwareSampler]],
    ) -> list[tuple[torch.Tensor, EntropyAwareSampler]]:
        """Stage 2: 中距離枝刈り。

        Stage 1の候補をL2トークンまで延長し、M_25でスコアリング後、
        上位B2個を選択する。

        Args:
            model: ARバックボーンモデル。
            candidates: Stage 1通過後の (候補テンソル, サンプラー) タプルのリスト。

        Returns:
            Stage 2を通過したB2個の (候補テンソル, サンプラー) タプルのリスト。
        """
        additional_tokens = self.L2 - self.L1

        # 各候補を (L2 - L1) トークン分延長
        extended: list[tuple[torch.Tensor, EntropyAwareSampler]] = []
        for candidate, sampler in candidates:
            ext = self._generate_n_tokens(model, candidate, additional_tokens, sampler)
            extended.append((ext, sampler))

        # M_25でスコアリング: 各候補の末尾L2トークンをバッチ化
        last_tokens = torch.stack([c[-self.L2:] for c, _ in extended])  # (B1, L2)
        with torch.no_grad():
            scores = self.detectors.score_mid(last_tokens)  # (B1,)

        # 上位B2個を選択
        top_indices = torch.argsort(scores, descending=True)[: self.B2]
        logger.info(
            "Stage 2 枝刈り: B1=%d → B2=%d (スコア上位: %s)",
            self.B1,
            self.B2,
            scores[top_indices].tolist(),
        )

        selected: list[tuple[torch.Tensor, EntropyAwareSampler]] = [
            extended[idx.item()] for idx in top_indices
        ]
        return selected

    # ------------------------------------------------------------------
    # Stage 3: 長スパン延長 (L3 = 50 トークン)
    # ------------------------------------------------------------------

    def _stage3_extend(
        self,
        model: object,
        candidates: list[tuple[torch.Tensor, EntropyAwareSampler]],
    ) -> list[torch.Tensor]:
        """Stage 3: 長スパン延長。

        Stage 2の候補をL3トークンまで延長する。

        Args:
            model: ARバックボーンモデル。
            candidates: Stage 2通過後の (候補テンソル, サンプラー) タプルのリスト。

        Returns:
            L3トークンまで延長されたB2個の候補テンソルのリスト。
        """
        additional_tokens = self.L3 - self.L2

        extended: list[torch.Tensor] = []
        for candidate, sampler in candidates:
            ext = self._generate_n_tokens(model, candidate, additional_tokens, sampler)
            extended.append(ext)

        logger.info("Stage 3 延長完了: %d 候補をL3=%dトークンまで延長", len(extended), self.L3)
        return extended

    # ------------------------------------------------------------------
    # 最終選択: マルチ解像度ランク集約
    # ------------------------------------------------------------------

    def _final_select(
        self, candidates: list[torch.Tensor]
    ) -> torch.Tensor:
        """最終選択: マルチ解像度ランク集約。

        M_50, M_50_25, M_50_10の3つの検出器でスコアリングし、
        重み付きランク集約により最良候補を選出する。

        Args:
            candidates: Stage 3完了後の候補テンソルのリスト。

        Returns:
            選出された最良候補の末尾L3トークン列 (L3,)。
        """
        # 各候補の末尾L3トークンをバッチ化
        last_tokens = torch.stack([c[-self.L3:] for c in candidates])  # (B2, L3)

        # 長スパン検出器3つでスコアリング
        with torch.no_grad():
            scores = self.detectors.score_long(last_tokens)

        # ランク重みを辞書に変換
        weights_dict = {
            "M_50": self.rank_weights[0],
            "M_50_25": self.rank_weights[1],
            "M_50_10": self.rank_weights[2],
        }

        # ランク集約により最良候補を選出
        best_idx = self.detectors.rank_aggregate(scores, weights_dict)
        logger.info(
            "最終選択: 候補 %d を選出 (スコア: M_50=%.4f, M_50_25=%.4f, M_50_10=%.4f)",
            best_idx.item(),
            scores["M_50"][best_idx].item(),
            scores["M_50_25"][best_idx].item(),
            scores["M_50_10"][best_idx].item(),
        )

        # 選出候補の末尾L3トークン（新規生成分）を返す
        return candidates[best_idx.item()][-self.L3:]

    # ------------------------------------------------------------------
    # メインエントリポイント
    # ------------------------------------------------------------------

    def generate(
        self,
        model: object,
        text_tokens: torch.Tensor,
        speaker_tokens: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """階層的デコーディングによるトークン列生成。

        ウォームアップ後、L3トークンごとに3ステージのビームサーチを
        反復的に実行し、EOSまたは最大長に達するまでトークンを生成する。

        Args:
            model: ARバックボーンモデル (HuggingFace CausalLM互換)。
            text_tokens: テキストトークン列 (text_len,)。
            speaker_tokens: 話者プロンプトトークン列 (speaker_len,)。
            eos_token_id: EOSトークンID。Noneの場合はEOS検出をスキップ。

        Returns:
            生成されたトークン列（プレフィックスを除く）。
        """
        # プレフィックスの構築
        prefix = torch.cat([text_tokens, speaker_tokens])
        prefix_len = prefix.size(0)

        # ウォームアップフェーズ
        sequence = self._warmup_phase(model, prefix)

        iteration = 0
        while sequence.size(0) < prefix_len + self.max_length:
            iteration += 1
            logger.info(
                "反復 %d: 現在の列長=%d (生成済み=%d / 最大=%d)",
                iteration,
                sequence.size(0),
                sequence.size(0) - prefix_len,
                self.max_length,
            )

            # Stage 1: 短スパン枝刈り
            stage1_candidates = self._stage1_prune(model, sequence)

            # Stage 2: 中距離枝刈り
            stage2_candidates = self._stage2_prune(model, stage1_candidates)

            # Stage 3: 長スパン延長
            stage3_candidates = self._stage3_extend(model, stage2_candidates)

            # 最終選択: マルチ解像度ランク集約
            best_tokens = self._final_select(stage3_candidates)

            # 選出されたL3トークンをシーケンスに追加
            sequence = torch.cat([sequence, best_tokens])

            # EASサンプラーの状態を最良候補のサンプラー状態に更新するため
            # 最終選択で選ばれた候補のサンプラーを追跡する必要がある。
            # ただし _final_select はテンソルのみ返すため、ここでは
            # eas_sampler のメモリを単純にリセットして次の反復に備える。
            # 各反復でフォーク元になるので、新しい反復の開始時点で
            # メモリバッファは前の反復から引き継がれなくてよい。

            # EOSトークンの検出
            if eos_token_id is not None and (best_tokens == eos_token_id).any():
                eos_pos = (best_tokens == eos_token_id).nonzero(as_tuple=True)[0][0]
                # EOS位置までのトークンを保持（EOSは含めない）
                # sequenceの末尾L3トークンがbest_tokensなので、
                # EOS以降を除去する
                tokens_to_remove = self.L3 - eos_pos.item()
                sequence = sequence[:-tokens_to_remove]
                logger.info("EOSトークンを検出。生成を終了します。")
                break

        # プレフィックスを除いた生成トークンを返す
        generated = sequence[prefix_len:]
        logger.info("生成完了: %d トークン生成", generated.size(0))
        return generated
