"""Vision inference pipelines and lazy-loaded model backends."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from PIL import Image

from immich_booru_tagger.config import (
    AnimePipelineConfig,
    PipelinesConfig,
    PipelineThresholdConfig,
    RoutingMode,
)

PIXAI_MODEL_NAME = "v0.9"
ANIME_RATING_REPO_ID = "animetimm/mobilenetv3_large_150d.dbv4-full"
ANIME_CLASSIFY_MODEL_NAME = "mobilenetv3_v1.3_dist"
ANIME_COMPLETENESS_MODEL_NAME = "mobilenetv3_v2.2_dist"
ANIME_REAL_MODEL_NAME = "mobilenetv3_v1.2_dist"
MAX_INFERENCE_EDGE = 640

logger = logging.getLogger(__name__)


@contextmanager
def _prepare_inference_image(
    image: Image.Image, *, max_dimension: int = MAX_INFERENCE_EDGE
) -> Iterator[Image.Image]:
    """Normalize image mode to RGB and cap long edge once for downstream models."""
    needs_resize = max(image.width, image.height) > max_dimension
    needs_rgb = image.mode != "RGB"
    if not needs_resize and not needs_rgb:
        yield image
        return

    prepared = image.convert("RGB") if needs_rgb else image.copy()
    try:
        if needs_resize:
            prepared.thumbnail((max_dimension, max_dimension), Image.Resampling.BILINEAR)
        yield prepared
    finally:
        prepared.close()

DANBOORU_RATINGS: tuple[str, ...] = (
    "general",
    "sensitive",
    "questionable",
    "explicit",
)
SANKAKU_RATINGS: tuple[str, ...] = (
    "safe",
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


def _select_top_rating(
    rating_scores: Mapping[str, float],
    allowed_levels: tuple[str, ...],
) -> str | None:
    filtered = {
        level: float(rating_scores[level])
        for level in allowed_levels
        if level in rating_scores
    }
    if not filtered:
        return None
    return max(filtered, key=lambda level: filtered[level])


class AnimePipeline:
    """Four-stage Anime Pipeline combining semantic tagging, rating, medium, and completeness."""

    def __init__(
        self,
        config: AnimePipelineConfig | None = None,
        *,
        backend_factory: Callable[[], AnimeBackend] | None = None,
    ) -> None:
        self._config = config or AnimePipelineConfig()
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

        with _prepare_inference_image(image) as prepared:
            general, character, copyright_tags = backend.extract_pixai(
                prepared,
                general_threshold=self._config.general_threshold,
                character_threshold=self._config.character_threshold,
            )
            tags.extend(("general", tag_name) for tag_name in general)
            tags.extend(("character", tag_name) for tag_name in character)
            tags.extend(("copyright", tag_name) for tag_name in copyright_tags)

            rating_scores = backend.extract_rating_scores(prepared)
            top_rating = _select_top_rating(rating_scores, DANBOORU_RATINGS)
            if top_rating is not None:
                tags.append(("rating", top_rating))

            if self._config.enable_medium_classification:
                medium_label, _ = backend.classify_medium(prepared)
                if medium_label and medium_label != "not_painting":
                    tags.append(("meta", medium_label))

            if self._config.enable_completeness_classification:
                completeness_label, _ = backend.classify_completeness(prepared)
                if completeness_label:
                    tags.append(("meta", completeness_label))

        return tags


class RealBackend(Protocol):
    """Low-level inference backend port for the Real Pipeline."""

    def extract_idolsankaku(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        """Return (rating_scores, general_scores, character_scores)."""
        ...


class RealPipeline:
    """Real-world photo tagging pipeline backed by IdolSankaku."""

    def __init__(
        self,
        thresholds: PipelineThresholdConfig | None = None,
        *,
        backend_factory: Callable[[], RealBackend] | None = None,
    ) -> None:
        self._thresholds = thresholds or PipelineThresholdConfig()
        self._backend_factory = backend_factory or get_shared_real_backend
        self._backend: RealBackend | None = None

    def _get_backend(self) -> RealBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def predict(self, image: Image.Image) -> list[tuple[str, str]]:
        """Predict Booru (category, tag_name) pairs for a real-world photo."""
        backend = self._get_backend()
        with _prepare_inference_image(image) as prepared:
            rating_scores, general, character = backend.extract_idolsankaku(
                prepared,
                general_threshold=self._thresholds.general_threshold,
                character_threshold=self._thresholds.character_threshold,
            )
        tags: list[tuple[str, str]] = []
        tags.extend(("general", tag_name) for tag_name in general)
        tags.extend(("character", tag_name) for tag_name in character)

        top_rating = _select_top_rating(rating_scores, SANKAKU_RATINGS)
        if top_rating is not None:
            tags.append(("rating", top_rating))
        return tags


_T = TypeVar("_T")


class _LazySingleton(Generic[_T]):
    """Thread-safe process-level lazy singleton holder."""

    def __init__(self, default_loader: Callable[[], _T]) -> None:
        self._default_loader = default_loader
        self._lock = threading.Lock()
        self._instance: _T | None = None

    def get(self, loader: Callable[[], _T] | None = None) -> _T:
        with self._lock:
            if self._instance is None:
                active_loader = loader or self._default_loader
                self._instance = active_loader()
            return self._instance

    def clear(self) -> None:
        with self._lock:
            self._instance = None


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


def _coerce_tag_score_triple(
    raw_result: Any, *, source: str
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if not isinstance(raw_result, tuple) or len(raw_result) < 3:
        raise PipelineError(
            f"Unexpected {source} output: {type(raw_result).__name__}"
        )
    return (
        _coerce_tag_scores(raw_result[0]),
        _coerce_tag_scores(raw_result[1]),
        _coerce_tag_scores(raw_result[2]),
    )


def _ensure_hf_token_env() -> None:
    """Propagate cached huggingface_hub CLI token to HF_TOKEN for hfutils clients."""
    if os.environ.get("HF_TOKEN", "").strip():
        return
    try:
        from huggingface_hub import get_token
    except ImportError:
        return
    if token := get_token():
        os.environ["HF_TOKEN"] = token


_DLL_DIR_HANDLES: list[object] = []
_ONNX_PREPARE_LOCK = threading.Lock()
_ONNX_RUNTIME_PREPARED = False


def _detect_onnx_provider() -> tuple[str | None, list[str]]:
    available: list[str] = []
    try:
        import onnxruntime

        get_available = getattr(onnxruntime, "get_available_providers", None)
        if callable(get_available):
            available = [str(p) for p in get_available()]
    except ImportError:
        pass

    try:
        from imgutils.utils.onnxruntime import get_onnx_provider

        return str(get_onnx_provider()), available
    except Exception:
        pass

    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider", available
    return (available[0] if available else None), available


def _preload_onnx_gpu_dlls() -> None:
    """Expose bundled NVIDIA wheel bin dirs and preload CUDA/cuDNN DLLs when present."""
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.is_dir():
        bin_dirs = [str(p) for p in nvidia_root.glob("*/bin") if p.is_dir()]
        if bin_dirs:
            current_path = os.environ.get("PATH", "")
            existing = set(current_path.split(os.pathsep)) if current_path else set()
            missing = [d for d in bin_dirs if d not in existing]
            if missing:
                os.environ["PATH"] = (
                    os.pathsep.join(missing) + os.pathsep + current_path
                    if current_path
                    else os.pathsep.join(missing)
                )
            add_dll_directory = getattr(os, "add_dll_directory", None)
            if callable(add_dll_directory):
                for bin_dir in bin_dirs:
                    try:
                        _DLL_DIR_HANDLES.append(add_dll_directory(bin_dir))
                    except OSError:
                        pass

    try:
        import onnxruntime
    except ImportError:
        return
    preload = getattr(onnxruntime, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception:
            pass


def _prepare_onnx_runtime() -> None:
    global _ONNX_RUNTIME_PREPARED
    with _ONNX_PREPARE_LOCK:
        if _ONNX_RUNTIME_PREPARED:
            return
        _ensure_hf_token_env()
        _preload_onnx_gpu_dlls()
        provider, available = _detect_onnx_provider()
        if provider is not None:
            logger.info(
                "ONNX Runtime execution provider: %s (available=%s)",
                provider,
                available,
            )
        _ONNX_RUNTIME_PREPARED = True


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
            import imgutils.generic as generic_module
            import imgutils.tagging as tagging_module
            import imgutils.validate as validate_module
        except ImportError as exc:
            raise PipelineDependencyError(
                "dghs-imgutils is required for anime vision inference; "
                "install the 'cpu' or 'gpu' optional extra."
            ) from exc

        _prepare_onnx_runtime()
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
        return _coerce_tag_score_triple(raw_result, source="get_pixai_tags")

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


class RealutilsRealBackend:
    """Lazy-loaded dghs-realutils adapter implementing RealBackend."""

    def __init__(self, *, tagging_module: Any) -> None:
        self._tagging = tagging_module

    @classmethod
    def load(cls) -> RealutilsRealBackend:
        """Lazily import dghs-realutils tagging module."""
        try:
            import realutils.tagging as tagging_module
        except ImportError as exc:
            raise PipelineDependencyError(
                "dghs-realutils is required for real vision inference; "
                "install the 'cpu' or 'gpu' optional extra."
            ) from exc

        _prepare_onnx_runtime()
        return cls(tagging_module=tagging_module)

    def extract_idolsankaku(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        raw_result = self._tagging.get_idolsankaku_tags(
            image,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            fmt=("rating", "general", "character"),
        )
        return _coerce_tag_score_triple(raw_result, source="get_idolsankaku_tags")


class DomainClassifier(Protocol):
    """Low-level domain classification port distinguishing anime vs real images."""

    def classify_domain(self, image: Image.Image) -> tuple[str, float]:
        """Return (label, score) where label is 'anime' or 'real'."""
        ...


class ImgutilsDomainClassifier:
    """Lazy-loaded dghs-imgutils adapter implementing DomainClassifier."""

    def __init__(self, *, validate_module: Any) -> None:
        self._validate = validate_module

    @classmethod
    def load(cls) -> ImgutilsDomainClassifier:
        """Lazily import dghs-imgutils validate module."""
        try:
            import imgutils.validate as validate_module
        except ImportError as exc:
            raise PipelineDependencyError(
                "dghs-imgutils is required for domain classification; "
                "install the 'cpu' or 'gpu' optional extra."
            ) from exc

        _prepare_onnx_runtime()
        return cls(validate_module=validate_module)

    def classify_domain(self, image: Image.Image) -> tuple[str, float]:
        label, score = self._validate.anime_real(
            image, model_name=ANIME_REAL_MODEL_NAME
        )
        return str(label), float(score)


_SHARED_ANIME_BACKEND: _LazySingleton[AnimeBackend] = _LazySingleton(
    ImgutilsAnimeBackend.load
)
_SHARED_REAL_BACKEND: _LazySingleton[RealBackend] = _LazySingleton(
    RealutilsRealBackend.load
)
_SHARED_DOMAIN_CLASSIFIER: _LazySingleton[DomainClassifier] = _LazySingleton(
    ImgutilsDomainClassifier.load
)


def clear_shared_anime_backend_cache() -> None:
    """Clear the process-level cached AnimeBackend singleton."""
    _SHARED_ANIME_BACKEND.clear()


def get_shared_anime_backend(
    loader: Callable[[], AnimeBackend] | None = None,
) -> AnimeBackend:
    """Return the process-level lazy-loaded AnimeBackend singleton."""
    return _SHARED_ANIME_BACKEND.get(loader)


def clear_shared_real_backend_cache() -> None:
    """Clear the process-level cached RealBackend singleton."""
    _SHARED_REAL_BACKEND.clear()


def get_shared_real_backend(
    loader: Callable[[], RealBackend] | None = None,
) -> RealBackend:
    """Return the process-level lazy-loaded RealBackend singleton."""
    return _SHARED_REAL_BACKEND.get(loader)


def clear_shared_domain_classifier_cache() -> None:
    """Clear the process-level cached DomainClassifier singleton."""
    _SHARED_DOMAIN_CLASSIFIER.clear()


def get_shared_domain_classifier(
    loader: Callable[[], DomainClassifier] | None = None,
) -> DomainClassifier:
    """Return the process-level lazy-loaded DomainClassifier singleton."""
    return _SHARED_DOMAIN_CLASSIFIER.get(loader)


def _resolve_factory(
    instance: _T | None,
    factory: Callable[[], _T] | None,
    default_factory: Callable[[], _T],
) -> Callable[[], _T]:
    if instance is not None:
        return lambda: instance
    return factory or default_factory


class DefaultVisionPipeline:
    """Default VisionPipeline routing assets to lazy-loaded domain pipelines."""

    def __init__(
        self,
        config: PipelinesConfig | None = None,
        *,
        anime_backend: AnimeBackend | None = None,
        anime_backend_factory: Callable[[], AnimeBackend] | None = None,
        real_backend: RealBackend | None = None,
        real_backend_factory: Callable[[], RealBackend] | None = None,
        domain_classifier: DomainClassifier | None = None,
        domain_classifier_factory: Callable[[], DomainClassifier] | None = None,
    ) -> None:
        self._config = config or PipelinesConfig()
        self._anime_backend_factory = _resolve_factory(
            anime_backend, anime_backend_factory, get_shared_anime_backend
        )
        self._real_backend_factory = _resolve_factory(
            real_backend, real_backend_factory, get_shared_real_backend
        )
        self._domain_classifier_factory = _resolve_factory(
            domain_classifier,
            domain_classifier_factory,
            get_shared_domain_classifier,
        )
        self._anime_pipeline: AnimePipeline | None = None
        self._real_pipeline: RealPipeline | None = None
        self._domain_classifier: DomainClassifier | None = None

    def _get_anime_pipeline(self) -> AnimePipeline:
        if self._anime_pipeline is None:
            self._anime_pipeline = AnimePipeline(
                self._config.anime,
                backend_factory=self._anime_backend_factory,
            )
        return self._anime_pipeline

    def _get_real_pipeline(self) -> RealPipeline:
        if self._real_pipeline is None:
            self._real_pipeline = RealPipeline(
                self._config.real,
                backend_factory=self._real_backend_factory,
            )
        return self._real_pipeline

    def _get_domain_classifier(self) -> DomainClassifier:
        if self._domain_classifier is None:
            self._domain_classifier = self._domain_classifier_factory()
        return self._domain_classifier

    def _run_domain(
        self, image: Image.Image, domain: Literal["anime", "real"]
    ) -> list[tuple[str, str]]:
        tags = (
            self._get_anime_pipeline().predict(image)
            if domain == "anime"
            else self._get_real_pipeline().predict(image)
        )
        if self._config.enable_domain_tag:
            tags.append(("meta", domain))
        return tags

    def predict(
        self, image: Image.Image, mode: RoutingMode
    ) -> Sequence[tuple[str, str]]:
        """Predict raw (category, tag_name) pairs for a decoded image under RoutingMode."""
        if mode not in ("anime", "real", "auto"):
            raise PipelineError(f"Unsupported routing mode: {mode!r}")

        with _prepare_inference_image(image) as prepared:
            if mode == "anime":
                return self._run_domain(prepared, "anime")
            if mode == "real":
                return self._run_domain(prepared, "real")
            domain, _ = self._get_domain_classifier().classify_domain(prepared)
            if domain == "anime":
                return self._run_domain(prepared, "anime")
            if domain == "real":
                return self._run_domain(prepared, "real")
            raise PipelineError(
                f"Unexpected domain classification label: {domain!r}"
            )
