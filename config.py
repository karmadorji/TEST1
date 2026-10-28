from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Project(StrictModel):
    name: str
    version: str
    seed: int


class Paths(StrictModel):
    labels: str
    cache_dir: str
    usage_dir: str
    eval_results: str


class ModelChoice(StrictModel):
    default: str
    alternative: str
    cpu_friendly: str | None = None


class SpeechModel(StrictModel):
    enabled: bool
    default: str


class VideoModel(StrictModel):
    frame_encoder: str
    frames: int = Field(ge=1, le=64)
    pooling: Literal["mean", "max"]


class Models(StrictModel):
    image: ModelChoice
    audio: ModelChoice
    speech: SpeechModel
    video: VideoModel


class Recognition(StrictModel):
    top_k: int = Field(ge=1, le=20)
    unknown_label: str
    uncertain_outcome: str
    prompt_aggregation: Literal["mean", "max", "sum"]
    temperature: float = Field(gt=0)


class Fusion(StrictModel):
    min_confidence: float = Field(ge=0, le=1)
    uncertain_below: float = Field(ge=0, le=1)
    weights: Dict[str, float]
    agreement_bonus: Dict[str, float]

    @model_validator(mode="after")
    def validate_fusion(self):
        expected = {"image", "audio", "video", "text"}
        if set(self.weights) != expected:
            raise ValueError(f"fusion.weights must contain exactly {sorted(expected)}")
        if any(v < 0 or v > 1 for v in self.weights.values()):
            raise ValueError("fusion weights must be in [0, 1]")
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("fusion weights must sum to 1.0")
        verdicts = {"MATCH", "PARTIAL", "CONFLICT", "SINGLE_MODALITY", "UNCERTAIN"}
        if set(self.agreement_bonus) != verdicts:
            raise ValueError("agreement_bonus must define every verdict")
        if any(v < 0 or v > 1 for v in self.agreement_bonus.values()):
            raise ValueError("agreement bonuses must be in [0, 1]")
        return self


class ImageMedia(StrictModel):
    min_width: int = Field(ge=1)
    min_height: int = Field(ge=1)
    max_side: int = Field(ge=1)
    blur_laplacian_variance_below: float = Field(ge=0)


class AudioMedia(StrictModel):
    sample_rate: int = Field(ge=8000)
    max_seconds: int = Field(ge=1)
    silence_rms_below: float = Field(ge=0)
    low_snr_db_below: float


class VideoMedia(StrictModel):
    max_seconds: int = Field(ge=1)
    frames: int = Field(ge=1, le=64)
    dark_mean_luma_below: float = Field(ge=0, le=255)
    shaky_motion_score_above: float = Field(ge=0)


class TextMedia(StrictModel):
    max_chars: int = Field(ge=1)


class Media(StrictModel):
    image: ImageMedia
    audio: AudioMedia
    video: VideoMedia
    text: TextMedia


class Reliability(StrictModel):
    request_timeout_seconds: int = Field(ge=1)
    cooldown_seconds: int = Field(ge=0)
    max_5xx_retries: int = Field(ge=0, le=10)
    backoff_base_seconds: float = Field(gt=0)
    rate_limit_per_minute: int = Field(ge=1)
    daily_soft_limit: int = Field(ge=1)


class Explanation(StrictModel):
    max_words: int = Field(ge=1)
    use_llm: bool
    template_fallback: bool


class Api(StrictModel):
    max_upload_mb: int = Field(ge=1)
    allowed_image_types: list[str]
    allowed_audio_types: list[str]
    allowed_video_types: list[str]


class AppConfig(StrictModel):
    project: Project
    paths: Paths
    models: Models
    recognition: Recognition
    fusion: Fusion
    media: Media
    reliability: Reliability
    explanation: Explanation
    api: Api


class Label(StrictModel):
    id: str
    display_name: str
    family: str
    prompts: list[str]
    reference_title: str
    kind: Literal["label", "outcome"] = "label"

    @model_validator(mode="after")
    def validate_prompts(self):
        if self.kind == "label" and not self.prompts:
            raise ValueError(f"label {self.id!r} must have at least one prompt")
        return self


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        return AppConfig.model_validate(yaml.safe_load(handle))


def load_labels(path: Path) -> list[Label]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    labels = [Label.model_validate(item) for item in raw]
    ids = [x.id for x in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("label ids must be unique")
    return labels


def validate(config_path: Path) -> tuple[AppConfig, list[Label]]:
    config = load_config(config_path)
    labels_path = (config_path.parent / config.paths.labels).resolve()
    labels = load_labels(labels_path)
    ids = {x.id for x in labels}
    if config.recognition.unknown_label not in ids:
        raise ValueError("configured unknown label is absent from labels.json")
    if config.recognition.uncertain_outcome not in ids:
        raise ValueError("configured uncertain outcome is absent from labels.json")
    uncertain = next(x for x in labels if x.id == config.recognition.uncertain_outcome)
    if uncertain.kind != "outcome":
        raise ValueError("uncertain must be kind='outcome'")
    if config.media.video.frames != config.models.video.frames:
        raise ValueError("media.video.frames and models.video.frames must match")
    return config, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate project configuration and label catalog.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config, labels = validate(args.config.resolve())
    known = sum(1 for x in labels if x.kind == "label" and x.id != config.recognition.unknown_label)
    print(f"OK: {config.project.name} v{config.project.version}")
    print(f"OK: {known} known labels + unknown + uncertain outcome")
    print(f"OK: fusion weights sum to {sum(config.fusion.weights.values()):.2f}")
    print(f"OK: image model = {config.models.image.default}")
    print(f"OK: audio model = {config.models.audio.default}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
