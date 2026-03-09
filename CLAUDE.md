# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSpoof-TTS: A training-free inference framework that improves discrete speech synthesis through hierarchical spoof-guided decoding. Based on the paper "Hierarchical Decoding for Discrete Speech Synthesis with Multi-Resolution Spoof Detection" (arXiv:2603.05373).

## Architecture

The system has two main components:

1. **Multi-Resolution Token-Level Spoof Detectors** — Five Conformer-based binary classifiers (M_50, M_25, M_10, M_50←25, M_50←10) that distinguish real vs synthetic codec token segments at different temporal resolutions. Shared architecture (d_model=256, 8 heads, 4 Transformer layers, FFN dim=1024), separate parameters per resolution.

2. **Hierarchical Spoof-Guided Sampling** — A two-part decoding strategy:
   - **Entropy-Aware Sampling (EAS)**: Token-level sampling with memory buffer tracking repetitions via inverse rank weighting and exponential temporal decay (Algorithm 1)
   - **Hierarchical Beam Search with Progressive Discriminator Pruning**: Multi-stage beam search where candidates are progressively pruned by short→mid→long span discriminators, then re-ranked via weighted rank aggregation (Algorithm 2)

## Key Dependencies

- **Base TTS**: NeuTTS (https://github.com/neuphonic/neutts) — pretrained codec-based language model, kept frozen
- **Speech Codec**: NeuCodec tokenizer for discrete token conversion
- **Conformer**: For spoof detector backbone (convolution-augmented transformer)
- **Datasets**: LibriTTS (training), LibriSpeech/LibriTTS/TwistList (evaluation)
- **Evaluation**: Whisper-large-v3 (WER), WavLM-base-plus-sv (speaker similarity), NISQA, MOSNet

## Key Hyperparameters

### Spoof Detector Training
- AdamW, lr=1e-4, weight_decay=1e-4, dropout=0.1, BCE loss
- Segment lengths: L ∈ {10, 25, 50}, skip-sample rates: r ∈ {2, 5}
- Single NVIDIA L40S GPU

### Decoding (EAS)
- top-k=50, top-p=0.8, temperature=1.0
- cluster_size k_e=3, memory_window W=15
- α=0.2, β=0.7, γ=0.8

### Hierarchical Decoding
- warmup L_w=20, stage_lengths (L1,L2,L3)=(10,25,50)
- beam_sizes (B0,B1,B2)=(8,5,3)
- rank_weights (w50,w25,w10) set equally

## Documentation

Detailed paper analysis is available in `docs/`:
- `docs/00_overview.md` — Full paper summary
- `docs/01_spoof_detection.md` — Multi-resolution spoof detection framework
- `docs/02_conformer_architecture.md` — Spoof detector model architecture
- `docs/03_entropy_aware_sampling.md` — EAS algorithm details
- `docs/04_hierarchical_decoding.md` — Hierarchical beam search with progressive pruning
- `docs/05_training.md` — Training data preparation and procedure
- `docs/06_evaluation.md` — Metrics, datasets, and results
- `docs/07_base_tts.md` — NeuTTS base system and codec details
- `docs/08_related_work.md` — Related work and positioning
- `docs/09_implementation_plan.md` — Reproduction roadmap and code structure
