"""End-to-end CLI tests driving cli.main with real TOML files and httpx.MockTransport."""

import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from PIL import Image

from immich_booru_tagger.cli import main
from immich_booru_tagger.config import RoutingMode
from immich_booru_tagger.pipeline import (
    clear_shared_anime_backend_cache,
    clear_shared_domain_classifier_cache,
    clear_shared_real_backend_cache,
)


class _NoopVisionPipeline:
    def predict(
        self, image: Image.Image, mode: RoutingMode
    ) -> Sequence[tuple[str, str]]:
        return ()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(0, 128, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def test_cli_dry_run_end_to_end_with_env_override_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://placeholder.local:2283"

[batch]
size = 1

[[albums]]
name = "Illustrations"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("IMMICH_BASE_URL", "https://immich.example.com")
    monkeypatch.setenv("IMMICH_API_KEY", "cli-env-key")

    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.url.path == "/api/albums":
            return _json_response(200, [{"id": "alb-1", "albumName": "Illustrations"}])
        if request.url.path == "/api/albums/alb-1":
            return _json_response(
                200,
                {
                    "id": "alb-1",
                    "albumName": "Illustrations",
                    "assets": [
                        {
                            "id": "a-done",
                            "type": "IMAGE",
                            "originalFileName": "done.png",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                            "tags": [{"id": "t1", "value": "meta/booru_tagged"}],
                        },
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "poster.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [],
                        },
                        {
                            "id": "a-2",
                            "type": "IMAGE",
                            "originalFileName": "banner.png",
                            "exifInfo": {"exifImageWidth": 2500, "exifImageHeight": 1000},
                            "tags": [],
                        },
                    ],
                },
            )
        if request.url.path.endswith("/thumbnail"):
            return httpx.Response(status_code=200, content=_png_bytes())
        return _json_response(404, {"message": "Not found"})

    exit_code = main(
        ["--config", str(cfg_file), "--dry-run", "--limit", "1"],
        transport=httpx.MockTransport(handler),
        vision_pipeline=_NoopVisionPipeline(),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert all(req.method == "GET" for req in seen_requests)
    assert all(req.headers["x-api-key"] == "cli-env-key" for req in seen_requests)
    assert all(req.url.host == "immich.example.com" for req in seen_requests)
    assert "poster.png" in captured.out
    assert "meta/absurdres" in captured.out
    assert "banner.png" not in captured.out


def test_cli_force_flag_includes_already_tagged_assets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-1"
""".strip(),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/albums/alb-1":
            return _json_response(
                200,
                {
                    "id": "alb-1",
                    "albumName": "Mixed",
                    "assets": [
                        {
                            "id": "a-done",
                            "type": "IMAGE",
                            "originalFileName": "retag_me.png",
                            "exifInfo": {"exifImageWidth": 1600, "exifImageHeight": 1200},
                            "tags": [{"id": "t1", "value": "meta/booru_tagged"}],
                        }
                    ],
                },
            )
        if request.url.path.endswith("/thumbnail"):
            return httpx.Response(status_code=200, content=_png_bytes())
        return _json_response(404, {})

    exit_code = main(
        ["-c", str(cfg_file), "--dry-run", "--force"],
        transport=httpx.MockTransport(handler),
        vision_pipeline=_NoopVisionPipeline(),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "retag_me.png" in captured.out
    assert "meta/highres" in captured.out


def test_cli_returns_nonzero_on_missing_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_missing = main(["--config", str(tmp_path / "nope.toml"), "--dry-run"])
    err_missing = capsys.readouterr().err
    assert exit_missing != 0
    assert err_missing.strip() != ""


def test_cli_returns_nonzero_with_candidate_uuids_on_ambiguous_album(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "dup.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
name = "SameName"
""".strip(),
        encoding="utf-8",
    )

    def dup_handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            [
                {"id": "uuid-111", "albumName": "SameName"},
                {"id": "uuid-222", "albumName": "SameName"},
            ],
        )

    exit_dup = main(
        ["-c", str(cfg_file), "--dry-run"],
        transport=httpx.MockTransport(dup_handler),
    )
    err_dup = capsys.readouterr().err
    assert exit_dup != 0
    assert "uuid-111" in err_dup
    assert "uuid-222" in err_dup


def test_cli_non_dry_run_with_force_writes_tags_and_cleans_old_managed_tags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[batch]
size = 2

[[albums]]
id = "alb-1"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    write_events: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/alb-1":
            return _json_response(
                200,
                {
                    "id": "alb-1",
                    "albumName": "Illustrations",
                    "assets": [
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "retag.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [
                                {"id": "old-m", "value": "meta/booru_tagged"},
                                {"id": "user-keep", "value": "my_favorite"},
                            ],
                        }
                    ],
                },
            )

        if request.url.path.endswith("/thumbnail"):
            return httpx.Response(status_code=200, content=_png_bytes())

        body = json.loads(request.content.decode("utf-8")) if request.content else None
        write_events.append((request.method, request.url.path, body))

        if request.method == "PUT" and request.url.path == "/api/tags":
            assert isinstance(body, dict)
            return _json_response(
                200,
                [{"id": f"new-{i}", "value": v} for i, v in enumerate(body["tags"])],
            )
        if request.method in {"DELETE", "PUT"} and request.url.path.endswith("/assets"):
            return httpx.Response(status_code=204)
        return _json_response(404, {})

    exit_code = main(
        ["-c", str(cfg_file), "--force"],
        transport=httpx.MockTransport(handler),
        vision_pipeline=_NoopVisionPipeline(),
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "dry_run=False" in out
    assert "[dry-run]" not in out
    assert write_events == [
        ("DELETE", "/api/tags/old-m/assets", {"ids": ["a-1"]}),
        ("GET", "/api/tags", None),
        ("PUT", "/api/tags", {"tags": ["meta/absurdres"]}),
        ("PUT", "/api/tags/assets", {"assetIds": ["a-1"], "tagIds": ["new-0", "old-m"]}),
    ]


def test_cli_uses_default_vision_pipeline_for_anime_album(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_shared_anime_backend_cache()
    cfg_file = tmp_path / "anime.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-anime"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    class FakeTIMM:
        def __init__(self, repo_id: str) -> None:
            self.repo_id = repo_id

        def predict(
            self,
            image: Image.Image,
            *,
            thresholds: float = 0.0,
            use_tag_thresholds: bool = False,
            fmt: tuple[str, ...] = ("rating",),
        ) -> Mapping[str, float]:
            return {"general": 0.9, "sensitive": 0.05, "questionable": 0.03, "explicit": 0.02}

    imgutils_mod = ModuleType("imgutils")
    tagging_mod = ModuleType("imgutils.tagging")
    tagging_mod.get_pixai_tags = (  # type: ignore[attr-defined]
        lambda image, **kwargs: ({"1girl": 0.95}, {"hatsune_miku": 0.92}, ["vocaloid"])
    )
    validate_mod = ModuleType("imgutils.validate")
    validate_mod.anime_classify = lambda image, **kwargs: ("illustration", 0.97)  # type: ignore[attr-defined]
    validate_mod.anime_completeness = lambda image, **kwargs: ("polished", 0.96)  # type: ignore[attr-defined]
    generic_mod = ModuleType("imgutils.generic")
    generic_mod.MultiLabelTIMMModel = FakeTIMM  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "imgutils", imgutils_mod)
    monkeypatch.setitem(sys.modules, "imgutils.tagging", tagging_mod)
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate_mod)
    monkeypatch.setitem(sys.modules, "imgutils.generic", generic_mod)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/albums/alb-anime":
            return _json_response(
                200,
                {
                    "id": "alb-anime",
                    "albumName": "Anime",
                    "assets": [
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "miku.png",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                            "tags": [],
                        }
                    ],
                },
            )
        if request.url.path == "/api/assets/a-1/thumbnail":
            return httpx.Response(status_code=200, content=_png_bytes())
        return _json_response(404, {})

    try:
        exit_code = main(
            ["-c", str(cfg_file), "--dry-run"],
            transport=httpx.MockTransport(handler),
        )
    finally:
        clear_shared_anime_backend_cache()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "general/1girl" in out
    assert "character/hatsune_miku" in out
    assert "copyright/vocaloid" in out
    assert "rating/general" in out
    assert "meta/illustration" in out
    assert "meta/polished" in out
    assert "meta/highres" in out


def test_cli_returns_nonzero_when_default_vision_dependency_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_shared_anime_backend_cache()
    cfg_file = tmp_path / "anime.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-anime"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setitem(sys.modules, "imgutils", None)
    monkeypatch.setitem(sys.modules, "imgutils.tagging", None)
    monkeypatch.setitem(sys.modules, "imgutils.validate", None)
    monkeypatch.setitem(sys.modules, "imgutils.generic", None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/albums/alb-anime":
            return _json_response(
                200,
                {
                    "id": "alb-anime",
                    "albumName": "Anime",
                    "assets": [
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "miku.png",
                            "exifInfo": None,
                            "tags": [],
                        }
                    ],
                },
            )
        if request.url.path == "/api/assets/a-1/thumbnail":
            return httpx.Response(status_code=200, content=_png_bytes())
        return _json_response(404, {})

    exit_code = main(
        ["-c", str(cfg_file), "--dry-run"],
        transport=httpx.MockTransport(handler),
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "dghs-imgutils" in err


def test_cli_uses_default_vision_pipeline_for_auto_mode_routing_real_and_anime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_shared_anime_backend_cache()
    clear_shared_real_backend_cache()
    clear_shared_domain_classifier_cache()
    cfg_file = tmp_path / "auto.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-auto"
""".strip(),
        encoding="utf-8",
    )

    class FakeTIMM:
        def __init__(self, repo_id: str) -> None:
            self.repo_id = repo_id

        def predict(
            self,
            image: Image.Image,
            *,
            thresholds: float = 0.0,
            use_tag_thresholds: bool = False,
            fmt: tuple[str, ...] = ("rating",),
        ) -> Mapping[str, float]:
            return {"general": 0.9, "sensitive": 0.05, "questionable": 0.03, "explicit": 0.02}

    imgutils_mod = ModuleType("imgutils")
    tagging_mod = ModuleType("imgutils.tagging")
    tagging_mod.get_pixai_tags = (  # type: ignore[attr-defined]
        lambda image, **kwargs: ({"1girl": 0.95}, {"hatsune_miku": 0.92}, ["vocaloid"])
    )
    validate_mod = ModuleType("imgutils.validate")
    validate_mod.anime_real = (  # type: ignore[attr-defined]
        lambda image, **kwargs: ("real", 0.99) if image.size == (2, 2) else ("anime", 0.98)
    )
    validate_mod.anime_classify = lambda image, **kwargs: ("illustration", 0.97)  # type: ignore[attr-defined]
    validate_mod.anime_completeness = lambda image, **kwargs: ("polished", 0.96)  # type: ignore[attr-defined]
    generic_mod = ModuleType("imgutils.generic")
    generic_mod.MultiLabelTIMMModel = FakeTIMM  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "imgutils", imgutils_mod)
    monkeypatch.setitem(sys.modules, "imgutils.tagging", tagging_mod)
    monkeypatch.setitem(sys.modules, "imgutils.validate", validate_mod)
    monkeypatch.setitem(sys.modules, "imgutils.generic", generic_mod)

    realutils_mod = ModuleType("realutils")
    real_tagging_mod = ModuleType("realutils.tagging")
    real_tagging_mod.get_idolsankaku_tags = (  # type: ignore[attr-defined]
        lambda image, **kwargs: (
            {"safe": 0.91, "questionable": 0.06, "explicit": 0.03},
            {"portrait": 0.88},
            {},
        )
    )
    monkeypatch.setitem(sys.modules, "realutils", realutils_mod)
    monkeypatch.setitem(sys.modules, "realutils.tagging", real_tagging_mod)

    anime_buf = io.BytesIO()
    Image.new("RGB", (3, 3)).save(anime_buf, format="PNG")
    anime_png = anime_buf.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/albums/alb-auto":
            return _json_response(
                200,
                {
                    "id": "alb-auto",
                    "albumName": "Auto Album",
                    "assets": [
                        {
                            "id": "r-1",
                            "type": "IMAGE",
                            "originalFileName": "photo.jpg",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                            "tags": [],
                        },
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "miku.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [],
                        },
                    ],
                },
            )
        if request.url.path == "/api/assets/r-1/thumbnail":
            return httpx.Response(status_code=200, content=_png_bytes())
        if request.url.path == "/api/assets/a-1/thumbnail":
            return httpx.Response(status_code=200, content=anime_png)
        return _json_response(404, {})

    try:
        exit_code = main(
            ["-c", str(cfg_file), "--dry-run"],
            transport=httpx.MockTransport(handler),
        )
    finally:
        clear_shared_anime_backend_cache()
        clear_shared_real_backend_cache()
        clear_shared_domain_classifier_cache()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "general/portrait" in out
    assert "rating/safe" in out
    assert "meta/highres" in out
    assert "character/hatsune_miku" in out
    assert "rating/general" in out
    assert "meta/absurdres" in out


def test_cli_since_and_until_filters_assets_before_detail_lookup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "time.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-time"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/api/albums/alb-time":
            return _json_response(
                200,
                {
                    "id": "alb-time",
                    "albumName": "Time Album",
                    "assets": [
                        {
                            "id": "a-old",
                            "type": "IMAGE",
                            "originalFileName": "old.png",
                            "createdAt": "2026-05-31T12:00:00Z",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                        },
                        {
                            "id": "a-mid",
                            "type": "IMAGE",
                            "originalFileName": "mid.png",
                            "createdAt": "2026-06-15T12:00:00Z",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                        },
                        {
                            "id": "a-new",
                            "type": "IMAGE",
                            "originalFileName": "future.png",
                            "createdAt": "2026-07-02T12:00:00Z",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                        },
                    ],
                },
            )
        if request.url.path == "/api/assets/a-mid":
            return _json_response(
                200,
                {
                    "id": "a-mid",
                    "type": "IMAGE",
                    "originalFileName": "mid.png",
                    "tags": [],
                },
            )
        if request.url.path == "/api/assets/a-mid/thumbnail":
            return httpx.Response(status_code=200, content=_png_bytes())
        return _json_response(404, {"message": "Unexpected request"})

    exit_code = main(
        ["-c", str(cfg_file), "--dry-run", "--since", "2026-06-01", "--until", "2026-06-30"],
        transport=httpx.MockTransport(handler),
        vision_pipeline=_NoopVisionPipeline(),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen_paths == [
        "/api/albums/alb-time",
        "/api/assets/a-mid",
        "/api/assets/a-mid/thumbnail",
    ]
    assert "mid.png" in captured.out
    assert "old.png" not in captured.out
    assert "future.png" not in captured.out


def test_cli_returns_nonzero_when_since_is_after_until(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_file = tmp_path / "time.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"

[[albums]]
id = "alb-time"
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(
        ["-c", str(cfg_file), "--dry-run", "--since", "2026-06-30", "--until", "2026-06-01"],
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "--since" in err and "--until" in err
