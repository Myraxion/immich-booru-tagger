"""Tests for VisionPipeline, AnimePipeline, and dghs-imgutils adapter."""

from __future__ import annotations

from collections.abc import Mapping
import sys
from types import ModuleType

import pytest
from PIL import Image

from immich_booru_tagger.config import PipelinesConfig, PipelineThresholdConfig
from immich_booru_tagger.pipeline import (
    ANIME_RATING_REPO_ID,
    AnimePipeline,
    DefaultVisionPipeline,
    PipelineDependencyError,
    clear_shared_anime_backend_cache,
    get_shared_anime_backend,
)


class _FakeAnimeBackend:
    def __init__(
        self,
        *,
        general: Mapping[str, float] | None = None,
        character: Mapping[str, float] | None = None,
        copyright_tags: Mapping[str, float] | None = None,
        rating_scores: Mapping[str, float] | None = None,
        medium: tuple[str, float] = ("illustration", 0.95),
        completeness: tuple[str, float] = ("polished", 0.91),
    ) -> None:
        self.general = dict(general or {})
        self.character = dict(character or {})
        self.copyright_tags = dict(copyright_tags or {})
        self.rating_scores = dict(rating_scores or {})
        self.medium = medium
        self.completeness = completeness
        self.pixai_calls: list[tuple[float, float]] = []

    def extract_pixai(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        self.pixai_calls.append((general_threshold, character_threshold))
        return self.general, self.character, self.copyright_tags

    def extract_rating_scores(self, image: Image.Image) -> Mapping[str, float]:
        return self.rating_scores

    def classify_medium(self, image: Image.Image) -> tuple[str, float]:
        return self.medium

    def classify_completeness(self, image: Image.Image) -> tuple[str, float]:
        return self.completeness


def test_anime_pipeline_combines_pixai_rating_medium_and_completeness() -> None:
    backend = _FakeAnimeBackend(
        general={"1girl": 0.98, "blue eyes": 0.72},
        character={"rem_(re:zero)": 0.93},
        copyright_tags={"re:zero_kara_hajimeru_isekai_seikatsu": 0.91},
        rating_scores={
            "general": 0.12,
            "sensitive": 0.79,
            "questionable": 0.07,
            "explicit": 0.02,
            "ignored_label": 0.99,
        },
        medium=("bangumi", 0.88),
        completeness=("polished", 0.94),
    )
    pipeline = AnimePipeline(
        PipelineThresholdConfig(general_threshold=0.4, character_threshold=0.9),
        backend_factory=lambda: backend,
    )
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    predicted = pipeline.predict(image)

    assert backend.pixai_calls == [(0.4, 0.9)]
    assert predicted == [
        ("general", "1girl"),
        ("general", "blue eyes"),
        ("character", "rem_(re:zero)"),
        ("copyright", "re:zero_kara_hajimeru_isekai_seikatsu"),
        ("rating", "sensitive"),
        ("meta", "bangumi"),
        ("meta", "polished"),
    ]
    assert all(category != "artist" for category, _ in predicted)


def test_anime_pipeline_ignores_not_painting_medium_label() -> None:
    backend = _FakeAnimeBackend(
        general={"monochrome": 0.85},
        rating_scores={"general": 0.95, "sensitive": 0.03, "questionable": 0.01, "explicit": 0.01},
        medium=("not_painting", 0.92),
        completeness=("monochrome", 0.89),
    )
    pipeline = AnimePipeline(backend_factory=lambda: backend)
    image = Image.new("RGB", (2, 2))

    predicted = pipeline.predict(image)

    assert predicted == [
        ("general", "monochrome"),
        ("rating", "general"),
        ("meta", "monochrome"),
    ]


def test_default_vision_pipeline_lazy_loads_shared_anime_backend_once() -> None:
    clear_shared_anime_backend_cache()
    init_count = 0

    def fake_loader() -> _FakeAnimeBackend:
        nonlocal init_count
        init_count += 1
        return _FakeAnimeBackend(
            general={"1girl": 0.9},
            rating_scores={"general": 0.9},
            medium=("illustration", 0.9),
            completeness=("polished", 0.9),
        )

    try:
        pipeline_a = DefaultVisionPipeline(
            PipelinesConfig(),
            anime_backend_factory=lambda: get_shared_anime_backend(fake_loader),
        )
        pipeline_b = DefaultVisionPipeline(
            PipelinesConfig(),
            anime_backend_factory=lambda: get_shared_anime_backend(fake_loader),
        )
        assert init_count == 0

        img = Image.new("RGB", (2, 2))
        first = pipeline_a.predict(img, "anime")
        second = pipeline_b.predict(img, "anime")

        assert init_count == 1
        assert first == second == [
            ("general", "1girl"),
            ("rating", "general"),
            ("meta", "illustration"),
            ("meta", "polished"),
        ]
    finally:
        clear_shared_anime_backend_cache()


@pytest.mark.parametrize("mode", ["auto", "real"])
def test_default_vision_pipeline_raises_not_implemented_for_other_modes_without_loading(
    mode: str,
) -> None:
    init_count = 0

    def fail_if_called() -> _FakeAnimeBackend:
        nonlocal init_count
        init_count += 1
        return _FakeAnimeBackend()

    pipeline = DefaultVisionPipeline(
        PipelinesConfig(),
        anime_backend_factory=fail_if_called,
    )
    img = Image.new("RGB", (2, 2))

    with pytest.raises(NotImplementedError):
        pipeline.predict(img, mode)  # type: ignore[arg-type]

    assert init_count == 0


def test_default_vision_pipeline_raises_dependency_error_when_imgutils_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_shared_anime_backend_cache()
    monkeypatch.setitem(sys.modules, "imgutils", None)
    monkeypatch.setitem(sys.modules, "imgutils.tagging", None)
    monkeypatch.setitem(sys.modules, "imgutils.validate", None)
    monkeypatch.setitem(sys.modules, "imgutils.generic", None)

    pipeline = DefaultVisionPipeline(PipelinesConfig())
    with pytest.raises(PipelineDependencyError, match="dghs-imgutils"):
        pipeline.predict(Image.new("RGB", (2, 2)), "anime")


def test_imgutils_anime_backend_contract_with_real_dghs_imgutils_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_shared_anime_backend_cache()
    recorded: dict[str, object] = {"timm_inits": []}

    def fake_get_pixai_tags(
        image: Image.Image,
        *,
        model_name: str = "v0.9",
        thresholds: Mapping[str, float] | None = None,
        fmt: tuple[str, ...] = ("general", "character", "ips", "ips_mapping"),
    ) -> tuple[dict[object, float], dict[str, float], list[str], dict[str, str]]:
        recorded["pixai_call"] = {
            "model_name": model_name,
            "thresholds": dict(thresholds or {}),
            "fmt": fmt,
        }
        return (
            {float("nan"): 0.99, "   ": 0.98, "1girl": 0.99},
            {"hakurei_reimu": 0.91},
            ["touhou"],
            {"hakurei_reimu": "touhou"},
        )

    class FakeMultiLabelTIMMModel:
        def __init__(self, repo_id: str) -> None:
            timm_inits = recorded["timm_inits"]
            assert isinstance(timm_inits, list)
            timm_inits.append(repo_id)

        def predict(
            self,
            image: Image.Image,
            *,
            thresholds: float | Mapping[str, float] | None = None,
            use_tag_thresholds: bool = True,
            fmt: tuple[str, ...] = ("general", "character", "rating"),
        ) -> tuple[dict[str, float]]:
            recorded["timm_predict"] = {
                "thresholds": thresholds,
                "use_tag_thresholds": use_tag_thresholds,
                "fmt": fmt,
            }
            return (
                {
                    "general": 0.05,
                    "sensitive": 0.81,
                    "questionable": 0.12,
                    "explicit": 0.02,
                },
            )

    def fake_anime_classify(
        image: Image.Image, *, model_name: str = "mobilenetv3_v1.3_dist"
    ) -> tuple[str, float]:
        recorded["classify_model"] = model_name
        return ("illustration", 0.96)

    def fake_anime_completeness(
        image: Image.Image, *, model_name: str = "mobilenetv3_v2.2_dist"
    ) -> tuple[str, float]:
        recorded["completeness_model"] = model_name
        return ("polished", 0.93)

    imgutils_mod = ModuleType("imgutils")
    tagging_mod = ModuleType("imgutils.tagging")
    tagging_mod.get_pixai_tags = fake_get_pixai_tags  # type: ignore[attr-defined]
    validate_mod = ModuleType("imgutils.validate")
    validate_mod.anime_classify = fake_anime_classify  # type: ignore[attr-defined]
    validate_mod.anime_completeness = fake_anime_completeness  # type: ignore[attr-defined]
    generic_mod = ModuleType("imgutils.generic")
    generic_mod.MultiLabelTIMMModel = FakeMultiLabelTIMMModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "imgutils", imgutils_mod)
    monkeypatch.setitem(sys.modules, "imgutils.tagging", tagging_mod)
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate_mod)
    monkeypatch.setitem(sys.modules, "imgutils.generic", generic_mod)

    try:
        config = PipelinesConfig(
            anime=PipelineThresholdConfig(general_threshold=0.33, character_threshold=0.87)
        )
        pipeline = DefaultVisionPipeline(config)
        img = Image.new("RGB", (4, 4))

        first = pipeline.predict(img, "anime")
        second = pipeline.predict(img, "anime")

        assert first == second == [
            ("general", "1girl"),
            ("character", "hakurei_reimu"),
            ("copyright", "touhou"),
            ("rating", "sensitive"),
            ("meta", "illustration"),
            ("meta", "polished"),
        ]
        assert recorded["timm_inits"] == [ANIME_RATING_REPO_ID]
        assert recorded["pixai_call"] == {
            "model_name": "v0.9",
            "thresholds": {"general": 0.33, "character": 0.87},
            "fmt": ("general", "character", "ips"),
        }
        assert recorded["timm_predict"] == {
            "thresholds": {"rating": 0.0},
            "use_tag_thresholds": False,
            "fmt": ("rating",),
        }
        assert recorded["classify_model"] == "mobilenetv3_v1.3_dist"
        assert recorded["completeness_model"] == "mobilenetv3_v2.2_dist"
    finally:
        clear_shared_anime_backend_cache()
