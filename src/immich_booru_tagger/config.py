"""TOML configuration schema, validation, and environment variable overrides."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

RoutingMode = Literal["auto", "anime", "real"]
ThumbnailSize = Literal["preview", "thumbnail"]


class ConfigError(ValueError):
    """Raised when the configuration file cannot be loaded or fails validation."""


class ImmichConfig(BaseModel):
    """Connection settings for the target Immich instance."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str
    thumbnail_size: ThumbnailSize = "preview"
    timeout: float = Field(default=30.0, gt=0.0)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must start with 'http://' or 'https://'")
        return normalized

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("api_key must not be empty")
        return stripped


class BatchConfig(BaseModel):
    """Micro-batch sizing configuration."""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=20, gt=0)


class PipelineThresholdConfig(BaseModel):
    """Confidence thresholds for a vision tagging pipeline."""

    model_config = ConfigDict(extra="forbid")

    general_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    character_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class PipelinesConfig(BaseModel):
    """Thresholds grouped by vision pipeline."""

    model_config = ConfigDict(extra="forbid")

    anime: PipelineThresholdConfig = Field(default_factory=PipelineThresholdConfig)
    real: PipelineThresholdConfig = Field(default_factory=PipelineThresholdConfig)


class AlbumConfig(BaseModel):
    """Target Album selector and routing mode."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str | None = None
    mode: RoutingMode = "auto"

    @model_validator(mode="after")
    def _validate_selector_exclusivity(self) -> AlbumConfig:
        has_id_key = self.id is not None
        has_name_key = self.name is not None
        if has_id_key == has_name_key:
            raise ValueError("Each album must specify exactly one of 'id' or 'name'")

        if self.id is not None:
            album_id = self.id.strip()
            if not album_id:
                raise ValueError("Album 'id' must not be empty")
            self.id = album_id
        if self.name is not None:
            album_name = self.name.strip()
            if not album_name:
                raise ValueError("Album 'name' must not be empty")
            self.name = album_name
        return self


class AppConfig(BaseModel):
    """Top-level validated application configuration."""

    model_config = ConfigDict(extra="forbid")

    immich: ImmichConfig
    batch: BatchConfig = Field(default_factory=BatchConfig)
    pipeline: PipelinesConfig = Field(default_factory=PipelinesConfig)
    albums: list[AlbumConfig] = Field(min_length=1)


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> AppConfig:
    """Load TOML config from path and apply IMMICH_API_KEY / IMMICH_BASE_URL overrides."""
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file {config_path}: {exc}") from exc

    try:
        raw_data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML syntax in {config_path}: {exc}") from exc

    environ = os.environ if env is None else env
    immich_section = raw_data.get("immich")
    if immich_section is None:
        immich_section = {}
        raw_data["immich"] = immich_section

    if isinstance(immich_section, dict):
        if api_key := environ.get("IMMICH_API_KEY", "").strip():
            immich_section["api_key"] = api_key
        if base_url := environ.get("IMMICH_BASE_URL", "").strip():
            immich_section["base_url"] = base_url

    try:
        return AppConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}: {exc}") from exc
