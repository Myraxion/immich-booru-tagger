"""Vision inference pipelines and lazy-loaded model backends."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from PIL import Image

from immich_booru_tagger.config import PipelinesConfig, PipelineThresholdConfig, RoutingMode

PIXAI_MODEL_NAME = "v0.9"
ANIME_RATING_REPO_ID = "animetimm/mobilenetv3_large_150d.dbv4-full"
ANIME_CLASSIFY_MODEL_NAME = "mobilenetv3_v1.3_dist"
ANIME_COMPLETENESS_MODEL_NAME = "mobilenetv3_v2.2_dist"

DANBOORU_RATINGS: tuple[str, ...] = (
    "general",
    "sensitive",
    "questionable",
    "explicit",
)


class PipelineError(RuntimeError):
    """Base error for vision pipeline execution failures."""


class PipelineDependencyError(PipelineError):
    """Raised when optional vision dependencies are missing."""


class VisionPipeline(Protocol):
    """Port for predicting raw (category, tag_name) pairs from an in-memory image."""

    def predict(
        self, image: Image.Image, mode: RoutingMode
    ) -> Sequence[tuple[str, str]]:
        """Predict raw (category, tag_name) pairs for a decoded image."""
        ...


class AnimeBackend(Protocol):
    """Low-level inference backend port for the four-stage Anime Pipeline."""

    def extract_pixai(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        """Return (general_scores, character_scores, copyright_scores)."""
        ...

    def extract_rating_scores(self, image: Image.Image) -> Mapping[str, float]:
        """Return raw Danbooru rating scores without threshold truncation."""
        ...

    def classify_medium(self, image: Image.Image) -> tuple[str, float]:
        """Return (label, score) from anime_classify."""
        ...

    def classify_completeness(self, image: Image.Image) -> tuple[str, float]:
        """Return (label, score) from anime_completeness."""
        ...


class AnimePipeline:
    """Four-stage Anime Pipeline combining semantic tagging, rating, medium, and completeness."""

    def __init__(
        self,
        thresholds: PipelineThresholdConfig | None = None,
        *,
        backend_factory: Callable[[], AnimeBackend] | None = None,
    ) -> None:
        self._thresholds = thresholds or PipelineThresholdConfig()
        self._backend_factory = backend_factory or get_shared_anime_backend
        self._backend: AnimeBackend | None = None

    def _get_backend(self) -> AnimeBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def predict(self, image: Image.Image) -> list[tuple[str, str]]:
        """Predict Booru (category, tag_name) pairs for an anime image."""
        backend = self._get_backend()
        tags: list[tuple[str, str]] = []

        general, character, copyright_tags = backend.extract_pixai(
            image,
            general_threshold=self._thresholds.general_threshold,
            character_threshold=self._thresholds.character_threshold,
        )
        tags.extend(("general", tag_name) for tag_name in general)
        tags.extend(("character", tag_name) for tag_name in character)
        tags.extend(("copyright", tag_name) for tag_name in copyright_tags)

        rating_scores = backend.extract_rating_scores(image)
        filtered_ratings = {
            level: float(rating_scores[level])
            for level in DANBOORU_RATINGS
            if level in rating_scores
        }
        if filtered_ratings:
            top_rating = max(filtered_ratings, key=lambda level: filtered_ratings[level])
            tags.append(("rating", top_rating))

        medium_label, _ = backend.classify_medium(image)
        if medium_label and medium_label != "not_painting":
            tags.append(("meta", medium_label))

        completeness_label, _ = backend.classify_completeness(image)
        if completeness_label:
            tags.append(("meta", completeness_label))

        return tags


_SHARED_ANIME_BACKEND_LOCK = threading.Lock()
_SHARED_ANIME_BACKEND: AnimeBackend | None = None


def _coerce_tag_scores(raw: Any) -> dict[str, float]:
    # Filter non-string keys (e.g. pandas NaN from empty tag CSV rows) and blank names.
    if isinstance(raw, Mapping):
        return {
            key: float(score)
            for key, score in raw.items()
            if isinstance(key, str) and key.strip()
        }
    if isinstance(raw, Sequence | set | frozenset) and not isinstance(raw, str | bytes):
        return {item: 1.0 for item in raw if isinstance(item, str) and item.strip()}
    return {}


class ImgutilsAnimeBackend:
    """Lazy-loaded dghs-imgutils adapter implementing AnimeBackend."""

    def __init__(
        self,
        *,
        tagging_module: Any,
        validate_module: Any,
        rating_model: Any,
    ) -> None:
        self._tagging = tagging_module
        self._validate = validate_module
        self._rating_model = rating_model

    @classmethod
    def load(cls) -> ImgutilsAnimeBackend:
        """Lazily import dghs-imgutils modules and initialize ONNX models."""
        try:
            import imgutils.generic as generic_module  # type: ignore[import-not-found]
            import imgutils.tagging as tagging_module  # type: ignore[import-not-found]
            import imgutils.validate as validate_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PipelineDependencyError(
                "dghs-imgutils is required for anime vision inference; "
                "install the 'cpu' or 'gpu' optional extra."
            ) from exc

        rating_model = generic_module.MultiLabelTIMMModel(ANIME_RATING_REPO_ID)
        return cls(
            tagging_module=tagging_module,
            validate_module=validate_module,
            rating_model=rating_model,
        )

    def extract_pixai(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        raw_result = self._tagging.get_pixai_tags(
            image,
            model_name=PIXAI_MODEL_NAME,
            thresholds={
                "general": general_threshold,
                "character": character_threshold,
            },
            fmt=("general", "character", "ips"),
        )
        if not isinstance(raw_result, tuple) or len(raw_result) < 3:
            raise PipelineError(
                f"Unexpected get_pixai_tags output: {type(raw_result).__name__}"
            )
        return (
            _coerce_tag_scores(raw_result[0]),
            _coerce_tag_scores(raw_result[1]),
            _coerce_tag_scores(raw_result[2]),
        )

    def extract_rating_scores(self, image: Image.Image) -> Mapping[str, float]:
        raw_result = self._rating_model.predict(
            image,
            thresholds={"rating": 0.0},
            use_tag_thresholds=False,
            fmt=("rating",),
        )
        if isinstance(raw_result, Mapping):
            return _coerce_tag_scores(raw_result)
        if isinstance(raw_result, tuple) and len(raw_result) == 1:
            return _coerce_tag_scores(raw_result[0])
        raise PipelineError(
            f"Unexpected MultiLabelTIMMModel rating output: {type(raw_result).__name__}"
        )

    def classify_medium(self, image: Image.Image) -> tuple[str, float]:
        label, score = self._validate.anime_classify(
            image, model_name=ANIME_CLASSIFY_MODEL_NAME
        )
        return str(label), float(score)

    def classify_completeness(self, image: Image.Image) -> tuple[str, float]:
        label, score = self._validate.anime_completeness(
            image, model_name=ANIME_COMPLETENESS_MODEL_NAME
        )
        return str(label), float(score)


def clear_shared_anime_backend_cache() -> None:
    """Clear the process-level cached AnimeBackend singleton."""
    global _SHARED_ANIME_BACKEND
    with _SHARED_ANIME_BACKEND_LOCK:
        _SHARED_ANIME_BACKEND = None


def get_shared_anime_backend(
    loader: Callable[[], AnimeBackend] | None = None,
) -> AnimeBackend:
    """Return the process-level lazy-loaded AnimeBackend singleton."""
    global _SHARED_ANIME_BACKEND
    with _SHARED_ANIME_BACKEND_LOCK:
        if _SHARED_ANIME_BACKEND is None:
            active_loader = loader or ImgutilsAnimeBackend.load
            _SHARED_ANIME_BACKEND = active_loader()
        return _SHARED_ANIME_BACKEND


class DefaultVisionPipeline:
    """Default VisionPipeline routing assets to lazy-loaded domain pipelines."""

    def __init__(
        self,
        config: PipelinesConfig | None = None,
        *,
        anime_backend: AnimeBackend | None = None,
        anime_backend_factory: Callable[[], AnimeBackend] | None = None,
    ) -> None:
        self._config = config or PipelinesConfig()
        if anime_backend is not None:
            self._anime_backend_factory: Callable[[], AnimeBackend] = (
                lambda: anime_backend
            )
        else:
            self._anime_backend_factory = (
                anime_backend_factory or get_shared_anime_backend
            )
        self._anime_pipeline: AnimePipeline | None = None

    def _get_anime_pipeline(self) -> AnimePipeline:
        if self._anime_pipeline is None:
            self._anime_pipeline = AnimePipeline(
                self._config.anime,
                backend_factory=self._anime_backend_factory,
            )
        return self._anime_pipeline

    def predict(
        self, image: Image.Image, mode: RoutingMode
    ) -> Sequence[tuple[str, str]]:
        """Predict raw (category, tag_name) pairs for a decoded image under RoutingMode."""
        if mode == "anime":
            return self._get_anime_pipeline().predict(image)
        raise NotImplementedError(
            f"Routing mode {mode!r} is not implemented yet; only 'anime' is supported"
        )
