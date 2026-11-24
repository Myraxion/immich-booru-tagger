"""Tests for VisionPipeline, AnimePipeline, and dghs-imgutils adapter."""

from __future__ import annotations

from collections.abc import Mapping
import sys
from types import ModuleType

import pytest
from PIL import Image

from immich_booru_tagger.config import (
    AnimePipelineConfig,
    PipelinesConfig,
    PipelineThresholdConfig,
)
from immich_booru_tagger.pipeline import (
    ANIME_RATING_REPO_ID,
    ANIME_REAL_MODEL_NAME,
    SANKAKU_RATINGS,
    AnimePipeline,
    DefaultVisionPipeline,
    PipelineDependencyError,
    PipelineError,
    RealPipeline,
    clear_shared_anime_backend_cache,
    clear_shared_domain_classifier_cache,
    clear_shared_real_backend_cache,
    get_shared_anime_backend,
    get_shared_domain_classifier,
    get_shared_real_backend,
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
        AnimePipelineConfig(general_threshold=0.4, character_threshold=0.9),
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
            ("meta", "anime"),
        ]
    finally:
        clear_shared_anime_backend_cache()


class _FakeDomainClassifier:
    def __init__(self, label_by_size: Mapping[tuple[int, int], str]) -> None:
        self._label_by_size = dict(label_by_size)
        self.calls: list[tuple[int, int]] = []

    def classify_domain(self, image: Image.Image) -> tuple[str, float]:
        self.calls.append(image.size)
        return (self._label_by_size[image.size], 0.99)


def test_default_vision_pipeline_forced_modes_skip_classifier_and_unused_backend() -> None:
    anime_init = 0
    real_init = 0
    classifier_init = 0

    def anime_loader() -> _FakeAnimeBackend:
        nonlocal anime_init
        anime_init += 1
        return _FakeAnimeBackend(general={"1girl": 0.9}, rating_scores={"general": 0.9})

    def real_loader() -> _FakeRealBackend:
        nonlocal real_init
        real_init += 1
        return _FakeRealBackend(rating_scores={"safe": 0.9}, general={"portrait": 0.8})

    def classifier_loader() -> _FakeDomainClassifier:
        nonlocal classifier_init
        classifier_init += 1
        return _FakeDomainClassifier({})

    anime_only = DefaultVisionPipeline(
        PipelinesConfig(),
        anime_backend_factory=anime_loader,
        real_backend_factory=real_loader,
        domain_classifier_factory=classifier_loader,
    )
    anime_only.predict(Image.new("RGB", (2, 2)), "anime")
    assert (anime_init, real_init, classifier_init) == (1, 0, 0)

    real_only = DefaultVisionPipeline(
        PipelinesConfig(),
        anime_backend_factory=anime_loader,
        real_backend_factory=real_loader,
        domain_classifier_factory=classifier_loader,
    )
    real_only.predict(Image.new("RGB", (2, 2)), "real")
    assert (anime_init, real_init, classifier_init) == (1, 1, 0)


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
            anime=AnimePipelineConfig(general_threshold=0.33, character_threshold=0.87)
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
            ("meta", "anime"),
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


class _FakeRealBackend:
    def __init__(
        self,
        *,
        rating_scores: Mapping[str, float] | None = None,
        general: Mapping[str, float] | None = None,
        character: Mapping[str, float] | None = None,
    ) -> None:
        self.rating_scores = dict(rating_scores or {})
        self.general = dict(general or {})
        self.character = dict(character or {})
        self.calls: list[tuple[float, float]] = []

    def extract_idolsankaku(
        self,
        image: Image.Image,
        *,
        general_threshold: float,
        character_threshold: float,
    ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
        self.calls.append((general_threshold, character_threshold))
        return self.rating_scores, self.general, self.character


def test_real_pipeline_extracts_general_character_and_top1_sankaku_rating() -> None:
    assert SANKAKU_RATINGS == ("safe", "questionable", "explicit")
    backend = _FakeRealBackend(
        rating_scores={
            "safe": 0.21,
            "questionable": 0.74,
            "explicit": 0.05,
            "sensitive": 0.99,  # Danbooru-only level must be ignored by Sankaku pipeline
        },
        general={"1girl": 0.91, "asian": 0.63},
        character={"some_idol": 0.89},
    )
    pipeline = RealPipeline(
        PipelineThresholdConfig(general_threshold=0.45, character_threshold=0.9),
        backend_factory=lambda: backend,
    )
    image = Image.new("RGB", (4, 4))

    predicted = pipeline.predict(image)

    assert backend.calls == [(0.45, 0.9)]
    assert predicted == [
        ("general", "1girl"),
        ("general", "asian"),
        ("character", "some_idol"),
        ("rating", "questionable"),
    ]


def test_real_pipeline_raises_dependency_error_when_realutils_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_shared_real_backend_cache()
    monkeypatch.setitem(sys.modules, "realutils", None)
    monkeypatch.setitem(sys.modules, "realutils.tagging", None)

    pipeline = RealPipeline()
    with pytest.raises(PipelineDependencyError, match="dghs-realutils"):
        pipeline.predict(Image.new("RGB", (2, 2)))


def test_realutils_real_backend_contract_and_shared_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_shared_real_backend_cache()
    recorded: list[dict[str, object]] = []

    def fake_get_idolsankaku_tags(
        image: Image.Image,
        *,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        fmt: tuple[str, ...] = ("rating", "general", "character"),
    ) -> tuple[dict[str, float], dict[object, float], dict[str, float]]:
        recorded.append(
            {
                "general_threshold": general_threshold,
                "character_threshold": character_threshold,
                "fmt": fmt,
            }
        )
        return (
            {"safe": 0.82, "questionable": 0.14, "explicit": 0.04},
            {float("nan"): 0.99, "  ": 0.95, "1girl": 0.88, "looking_at_viewer": 0.61},
            {"idol_name": 0.91},
        )

    realutils_mod = ModuleType("realutils")
    tagging_mod = ModuleType("realutils.tagging")
    tagging_mod.get_idolsankaku_tags = fake_get_idolsankaku_tags  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "realutils", realutils_mod)
    monkeypatch.setitem(sys.modules, "realutils.tagging", tagging_mod)

    try:
        shared_a = get_shared_real_backend()
        shared_b = get_shared_real_backend()
        assert shared_a is shared_b

        pipeline = RealPipeline(
            PipelineThresholdConfig(general_threshold=0.38, character_threshold=0.82)
        )
        tags = pipeline.predict(Image.new("RGB", (4, 4)))

        assert tags == [
            ("general", "1girl"),
            ("general", "looking_at_viewer"),
            ("character", "idol_name"),
            ("rating", "safe"),
        ]
        assert recorded == [
            {
                "general_threshold": 0.38,
                "character_threshold": 0.82,
                "fmt": ("rating", "general", "character"),
            }
        ]
    finally:
        clear_shared_real_backend_cache()


def test_default_vision_pipeline_real_mode_skips_anime_and_loads_real_lazily() -> None:
    anime_init_count = 0
    real_init_count = 0

    def fail_anime_loader() -> _FakeAnimeBackend:
        nonlocal anime_init_count
        anime_init_count += 1
        return _FakeAnimeBackend()

    fake_real = _FakeRealBackend(
        rating_scores={"safe": 0.1, "questionable": 0.3, "explicit": 0.6},
        general={"photo": 0.9},
        character={"idol_a": 0.92},
    )

    def real_loader() -> _FakeRealBackend:
        nonlocal real_init_count
        real_init_count += 1
        return fake_real

    pipeline = DefaultVisionPipeline(
        PipelinesConfig(
            real=PipelineThresholdConfig(general_threshold=0.5, character_threshold=0.8)
        ),
        anime_backend_factory=fail_anime_loader,
        real_backend_factory=real_loader,
    )
    assert real_init_count == 0

    tags = pipeline.predict(Image.new("RGB", (4, 4)), "real")

    assert anime_init_count == 0
    assert real_init_count == 1
    assert fake_real.calls == [(0.5, 0.8)]
    assert tags == [
        ("general", "photo"),
        ("character", "idol_a"),
        ("rating", "explicit"),
        ("meta", "real"),
    ]


def test_default_vision_pipeline_auto_mode_routes_and_lazy_loads_only_needed_pipeline() -> None:
    anime_init = 0
    real_init = 0

    def anime_loader() -> _FakeAnimeBackend:
        nonlocal anime_init
        anime_init += 1
        return _FakeAnimeBackend(
            general={"1girl": 0.9},
            rating_scores={"general": 0.9},
            medium=("illustration", 0.9),
            completeness=("polished", 0.9),
        )

    def real_loader() -> _FakeRealBackend:
        nonlocal real_init
        real_init += 1
        return _FakeRealBackend(
            rating_scores={"safe": 0.95},
            general={"outdoors": 0.8},
        )

    classifier = _FakeDomainClassifier({(2, 2): "anime", (3, 3): "real"})
    pipeline = DefaultVisionPipeline(
        PipelinesConfig(),
        anime_backend_factory=anime_loader,
        real_backend_factory=real_loader,
        domain_classifier=classifier,
    )

    anime_tags = pipeline.predict(Image.new("RGB", (2, 2)), "auto")
    assert anime_init == 1
    assert real_init == 0
    assert ("rating", "general") in anime_tags
    assert ("meta", "anime") in anime_tags

    real_tags = pipeline.predict(Image.new("RGB", (3, 3)), "auto")
    assert anime_init == 1
    assert real_init == 1
    assert real_tags == [("general", "outdoors"), ("rating", "safe"), ("meta", "real")]
    assert classifier.calls == [(2, 2), (3, 3)]


def test_default_vision_pipeline_auto_mode_raises_on_unknown_domain_label() -> None:
    classifier = _FakeDomainClassifier({(2, 2): "sketch"})
    pipeline = DefaultVisionPipeline(
        PipelinesConfig(),
        domain_classifier=classifier,
    )
    with pytest.raises(PipelineError, match="sketch"):
        pipeline.predict(Image.new("RGB", (2, 2)), "auto")


def test_imgutils_domain_classifier_contract_and_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_shared_domain_classifier_cache()
    monkeypatch.setitem(sys.modules, "imgutils", None)
    monkeypatch.setitem(sys.modules, "imgutils.validate", None)

    with pytest.raises(PipelineDependencyError, match="dghs-imgutils"):
        get_shared_domain_classifier().classify_domain(Image.new("RGB", (2, 2)))

    recorded_models: list[str] = []

    def fake_anime_real(
        image: Image.Image, *, model_name: str = "mobilenetv3_v1.2_dist"
    ) -> tuple[str, float]:
        recorded_models.append(model_name)
        return ("real", 0.98) if image.size == (3, 3) else ("anime", 0.99)

    imgutils_mod = ModuleType("imgutils")
    validate_mod = ModuleType("imgutils.validate")
    validate_mod.anime_real = fake_anime_real  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imgutils", imgutils_mod)
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate_mod)

    try:
        clear_shared_domain_classifier_cache()
        shared_a = get_shared_domain_classifier()
        shared_b = get_shared_domain_classifier()
        assert shared_a is shared_b
        assert shared_a.classify_domain(Image.new("RGB", (3, 3))) == ("real", 0.98)
        assert shared_a.classify_domain(Image.new("RGB", (2, 2))) == ("anime", 0.99)
        assert recorded_models == [ANIME_REAL_MODEL_NAME, ANIME_REAL_MODEL_NAME]
        assert ANIME_REAL_MODEL_NAME == "mobilenetv3_v1.2_dist"
    finally:
        clear_shared_domain_classifier_cache()


def test_anime_pipeline_skips_medium_and_completeness_when_disabled() -> None:
    class RecordingAnimeBackend(_FakeAnimeBackend):
        def __init__(self) -> None:
            super().__init__(
                general={"1girl": 0.9},
                rating_scores={"general": 0.9},
                medium=("illustration", 0.95),
                completeness=("polished", 0.91),
            )
            self.medium_called = False
            self.completeness_called = False

        def classify_medium(self, image: Image.Image) -> tuple[str, float]:
            self.medium_called = True
            return super().classify_medium(image)

        def classify_completeness(self, image: Image.Image) -> tuple[str, float]:
            self.completeness_called = True
            return super().classify_completeness(image)

    img = Image.new("RGB", (2, 2))

    backend_no_medium = RecordingAnimeBackend()
    tags_no_medium = AnimePipeline(
        AnimePipelineConfig(enable_medium_classification=False),
        backend_factory=lambda: backend_no_medium,
    ).predict(img)
    assert backend_no_medium.medium_called is False
    assert backend_no_medium.completeness_called is True
    assert tags_no_medium == [("general", "1girl"), ("rating", "general"), ("meta", "polished")]

    backend_no_completeness = RecordingAnimeBackend()
    tags_no_completeness = AnimePipeline(
        AnimePipelineConfig(enable_completeness_classification=False),
        backend_factory=lambda: backend_no_completeness,
    ).predict(img)
    assert backend_no_completeness.medium_called is True
    assert backend_no_completeness.completeness_called is False
    assert tags_no_completeness == [
        ("general", "1girl"),
        ("rating", "general"),
        ("meta", "illustration"),
    ]


def test_default_vision_pipeline_omits_domain_tag_when_disabled() -> None:
    anime_backend = _FakeAnimeBackend(
        general={"1girl": 0.9},
        rating_scores={"general": 0.9},
        medium=("illustration", 0.9),
        completeness=("polished", 0.9),
    )
    real_backend = _FakeRealBackend(
        rating_scores={"safe": 0.9},
        general={"portrait": 0.8},
    )
    classifier = _FakeDomainClassifier({(2, 2): "anime", (3, 3): "real"})
    pipeline = DefaultVisionPipeline(
        PipelinesConfig(enable_domain_tag=False),
        anime_backend=anime_backend,
        real_backend=real_backend,
        domain_classifier=classifier,
    )

    assert ("meta", "anime") not in pipeline.predict(Image.new("RGB", (2, 2)), "anime")
    assert ("meta", "real") not in pipeline.predict(Image.new("RGB", (3, 3)), "real")
    assert ("meta", "anime") not in pipeline.predict(Image.new("RGB", (2, 2)), "auto")
    assert ("meta", "real") not in pipeline.predict(Image.new("RGB", (3, 3)), "auto")
