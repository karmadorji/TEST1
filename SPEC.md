# Smart Multimodal Object and Environment Recognition System

## Scope
A free-first multimodal recognition demo that accepts any subset of image, audio, video, and optional text. It recognizes a fixed catalog of common objects/environments, rejects weak/out-of-scope evidence, detects cross-modal conflicts, and produces a deterministic fused verdict plus a short evidence-grounded explanation.

## Modalities
- **Image:** appearance, objects, animals, vehicles, people, electronics, household items, and scene context.
- **Audio:** non-speech events associated with labels (for example barking, meowing, engines, aircraft) and general acoustic context. Speech transcription is optional and is not itself classification evidence unless mapped to the fixed label catalog.
- **Video:** adds temporal evidence over still images by sampling multiple frames; its audio track is analyzed by the audio branch. Temporal aggregation reduces dependence on a single bad frame.
- **Optional text:** user-supplied contextual evidence/question. It may support or query fixed-label evidence but may not invent new labels or override deterministic fusion.

## Task type
Primary task: **recognize/classify with open-set rejection**. Secondary task: **object detection** for important objects in images/video. Simple media Q&A is constrained to evidence already returned by recognition/detection agents.

Generation mode is out of scope because the requested system recognizes and explains rather than creates media.

## Production backbones
- Image classification/embeddings: `openai/clip-vit-large-patch14`; CPU-friendly optional profile: `openai/clip-vit-base-patch32`.
- Audio event classification/embeddings: `laion/larger_clap_general`.
- Speech: Whisper family, disabled by default for the first recognition checkpoint.
- Video: uniformly sampled frames -> image encoder -> temporal mean/max support; extracted audio -> audio branch.
- Detection: optional open-vocabulary detector in a later checkpoint; detector outputs are mapped to the fixed catalog before fusion.

The technique ladder is zero-shot -> frozen embeddings + linear probe -> limited fine-tuning only when measured evaluation misses targets.

## Label catalog
`data/labels.json` is the only source of recognized labels. It includes known labels, `other/unknown`, and the non-label outcome `uncertain`. Each known label has a family, zero-shot prompts, and a reference title. The explainer may only use labels present in this catalog.

## Open-set behavior
A prediction is rejected when known-label support is weak, input quality is insufficient, or modalities conflict. `other/unknown` represents a valid out-of-catalog input. `uncertain` represents insufficient evidence to name a label. Thresholds are configuration values and must be tuned on held-out known and unknown examples.

## Success criteria
These are targets, not claimed measurements:

| Metric | Target |
|---|---:|
| Image known-label top-1 accuracy | >= 80% on curated held-out set |
| Audio known-label top-1 accuracy (labels with meaningful sounds) | >= 75% |
| Video known-label top-1 accuracy | >= 82% |
| Unknown rejection rate | >= 85% |
| Known-label false rejection rate | <= 15% |
| Mismatched image/audio -> `CONFLICT` | >= 90% |
| Hallucination rate on unknown groups | <= 10% |
| Fusion determinism | 100% identical for identical structured inputs/config |
| p95 fusion-only latency | < 50 ms on CPU |
| p95 end-to-end image latency | < 8 s on a reference 4-core CPU after model warm-up |
| p95 end-to-end image latency on free Colab GPU | < 3 s after warm-up |
| API failure isolation | 100%: one failed modality never crashes whole request |

Hardware-dependent latency targets are verified and reported with hardware details; they are never reported as achieved without measurement.

## Fusion contract
Usable modality = known top-1 label at or above `fusion.min_confidence`. Verdicts are `MATCH`, `PARTIAL`, `CONFLICT`, `SINGLE_MODALITY`, `UNCERTAIN`. Fusion is pure/deterministic. It uses configured modality weights, label support, and verdict bonuses. Below `fusion.uncertain_below`, no final known label is named.

## Quality controls
- Image: reject tiny dimensions; downscale large inputs; flag low resolution and blur using Laplacian variance.
- Audio: resample; cap duration; reject near-silence; estimate/flag low SNR.
- Video: cap duration; sample fixed frames; flag darkness and excessive inter-frame motion; extract audio when present.
- Text: length limit and normalization; never treated as calibrated probability.

## Likely failure modes
1. Blurred, dark, tiny, clipped, silent, noisy, or corrupt media.
2. Several equally salient objects when a single final label is expected.
3. Labels that are visually similar (bus/truck, cup/bottle) or acoustically weak (chair/table/tree).
4. Audio and visual evidence refer to different subjects, requiring `CONFLICT` rather than forced fusion.
5. Out-of-scope species/objects/scenes that resemble supported labels.
6. CLIP/CLAP prompt sensitivity and dataset bias; prompt ensembles reduce but do not eliminate this.
7. CPU memory/latency pressure from large backbones; the configuration provides a lighter image profile.
8. Missing modalities; fusion must degrade to `SINGLE_MODALITY` or `UNCERTAIN` without failure.

## Evaluation plan
Use fixed hash-based train/validation/test assignment. Report per-modality accuracy, confusion matrices, unknown rejection, false rejection, hallucination by unknown group, mismatched-pair conflict rate, latency, cache hit rate, and provider fallback rate. Store measured outputs under `eval/results/`; do not overwrite targets with unmeasured claims.
