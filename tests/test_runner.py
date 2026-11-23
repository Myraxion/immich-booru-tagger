"""End-to-end tests for BatchRunner through ImmichClient + httpx.MockTransport."""

import io
import json
import logging
from collections.abc import Mapping, Sequence

import httpx
import pytest
from PIL import Image

from immich_booru_tagger.config import (
    AlbumConfig,
    AppConfig,
    BatchConfig,
    ImmichConfig,
    PipelinesConfig,
    PipelineThresholdConfig,
    RoutingMode,
)
from immich_booru_tagger.immich import ImmichApiError, ImmichClient
from immich_booru_tagger.pipeline import DefaultVisionPipeline
from immich_booru_tagger.runner import BatchRunner


def _png_bytes(width: int = 1, height: int = 1, color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def _sample_config(*, batch_size: int = 2) -> AppConfig:
    return AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        batch=BatchConfig(size=batch_size),
        albums=[
            AlbumConfig(name="Anime Album", mode="anime"),
            AlbumConfig(id="album-real", mode="real"),
        ],
    )


def _sample_transport(requests_log: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests_log.append(request)
        if request.method == "GET" and request.url.path == "/api/albums":
            return _json_response(
                200,
                [{"id": "album-anime", "albumName": "Anime Album"}],
            )
        if request.method == "GET" and request.url.path == "/api/albums/album-anime":
            return _json_response(
                200,
                {
                    "id": "album-anime",
                    "albumName": "Anime Album",
                    "assets": [
                        {
                            "id": "img-tagged",
                            "type": "IMAGE",
                            "originalFileName": "already.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [{"id": "t-1", "value": "meta/booru_tagged"}],
                        },
                        {
                            "id": "vid-1",
                            "type": "VIDEO",
                            "originalFileName": "movie.mp4",
                            "exifInfo": {"exifImageWidth": 1920, "exifImageHeight": 1080},
                            "tags": [],
                        },
                        {
                            "id": "img-1",
                            "type": "IMAGE",
                            "originalFileName": "illu_absurd.png",
                            "exifInfo": {"exifImageWidth": 4000, "exifImageHeight": 3000},
                            "tags": [{"id": "t-user", "value": "favorites/2026"}],
                        },
                        {
                            "id": "img-2",
                            "type": "IMAGE",
                            "originalFileName": "tall_low.png",
                            "exifInfo": {"exifImageWidth": 200, "exifImageHeight": 500},
                            "tags": [],
                        },
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/api/albums/album-real":
            return _json_response(
                200,
                {
                    "id": "album-real",
                    "albumName": "Real Photos",
                    "assets": [
                        {
                            "id": "img-2",
                            "type": "IMAGE",
                            "originalFileName": "tall_low.png",
                            "exifInfo": {"exifImageWidth": 200, "exifImageHeight": 500},
                            "tags": [],
                        },
                        {
                            "id": "img-3",
                            "type": "IMAGE",
                            "originalFileName": "no_exif.jpg",
                            "exifInfo": None,
                            "tags": [],
                        },
                    ],
                },
            )
        if request.method == "GET" and request.url.path.startswith("/api/assets/") and request.url.path.endswith("/thumbnail"):
            size = (2, 2) if "/img-3/" in request.url.path else (1, 1)
            return httpx.Response(
                status_code=200,
                headers={"content-type": "image/png"},
                content=_png_bytes(*size),
            )
        return _json_response(404, {"message": "Not found"})

    return httpx.MockTransport(handler)


def test_batch_runner_filters_eligible_assets_splits_batches_and_dry_runs() -> None:
    requests_log: list[httpx.Request] = []
    config = _sample_config(batch_size=2)

    with ImmichClient.from_config(
        config, transport=_sample_transport(requests_log)
    ) as client:
        runner = BatchRunner(config=config, client=client)
        summary = runner.run(dry_run=True)

    assert all(req.method == "GET" for req in requests_log)
    assert summary.dry_run is True
    assert summary.albums_resolved == 2
    assert summary.total_eligible == 3
    assert summary.processed_count == 3
    assert summary.batches_count == 2

    by_id = {result.asset_id: result for result in summary.results}
    assert list(by_id.keys()) == ["img-1", "img-2", "img-3"]
    assert by_id["img-1"].tags == ["meta/absurdres"]
    assert by_id["img-1"].mode == "anime"
    assert by_id["img-2"].tags == ["meta/lowres", "meta/tall_image"]
    assert by_id["img-3"].tags == []
    assert by_id["img-3"].mode == "real"


def test_batch_runner_applies_limit_and_force_flags() -> None:
    requests_log: list[httpx.Request] = []
    config = _sample_config(batch_size=10)

    with ImmichClient.from_config(
        config, transport=_sample_transport(requests_log)
    ) as client:
        runner = BatchRunner(config=config, client=client)
        summary = runner.run(dry_run=True, limit=2, force=True)

    assert summary.total_eligible == 4
    assert summary.processed_count == 2
    assert summary.batches_count == 1
    assert [result.asset_id for result in summary.results] == ["img-tagged", "img-1"]
    assert summary.results[0].tags == ["meta/absurdres"]


def test_batch_runner_merges_and_normalizes_vision_pipeline_tags_with_dedup() -> None:
    class FakeVisionPipeline:
        def predict(
            self, image: Image.Image, mode: RoutingMode
        ) -> Sequence[tuple[str, str]]:
            return [
                ("copyright", "Fate/Grand Order"),
                ("character", "Rem (Re:Zero)"),
                ("meta", "absurdres"),  # overlaps with EXIF meta/absurdres
                ("general", "1girl"),
                ("general", "1girl"),  # duplicate from vision
            ]

    requests_log: list[httpx.Request] = []
    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        batch=BatchConfig(size=5),
        albums=[AlbumConfig(id="album-anime", mode="anime")],
    )

    with ImmichClient.from_config(
        config, transport=_sample_transport(requests_log)
    ) as client:
        runner = BatchRunner(
            config=config, client=client, vision_pipeline=FakeVisionPipeline()
        )
        summary = runner.run(dry_run=True, limit=1)

    assert len(summary.results) == 1
    assert summary.results[0].tags == [
        "copyright/fate_grand_order",
        "character/rem_(re:zero)",
        "meta/absurdres",
        "general/1girl",
    ]


def test_batch_runner_rejects_non_positive_limit() -> None:
    requests_log: list[httpx.Request] = []
    config = _sample_config()

    with ImmichClient.from_config(
        config, transport=_sample_transport(requests_log)
    ) as client:
        runner = BatchRunner(config=config, client=client)

        with pytest.raises(ValueError):
            runner.run(dry_run=True, limit=0)


def test_batch_runner_writes_back_per_micro_batch_reusing_tag_cache() -> None:
    class FakeVisionPipeline:
        def predict(
            self, image: Image.Image, mode: RoutingMode
        ) -> Sequence[tuple[str, str]]:
            if image.size == (2, 2):
                return [("general", "1girl"), ("rating", "safe")]
            return [("general", "1girl")]

    write_events: list[tuple[str, str, object]] = []
    next_tag_id = 0
    tag_registry: dict[str, str] = {}

    base_transport = _sample_transport([])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/api/tags":
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            created = []
            nonlocal next_tag_id
            for value in body["tags"]:
                if value not in tag_registry:
                    next_tag_id += 1
                    tag_registry[value] = f"tid-{next_tag_id}"
                created.append({"id": tag_registry[value], "value": value})
            return _json_response(200, created)

        if request.method == "PUT" and request.url.path.endswith("/assets"):
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            return httpx.Response(status_code=204)

        return base_transport.handle_request(request)

    config = _sample_config(batch_size=2)

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(
            config=config, client=client, vision_pipeline=FakeVisionPipeline()
        )
        summary = runner.run(dry_run=False)

    assert summary.dry_run is False
    assert summary.processed_count == 3
    assert summary.batches_count == 2

    # First micro-batch: img-1 & img-2; second micro-batch: img-3
    id_1girl = tag_registry["general/1girl"]
    id_absurdres = tag_registry["meta/absurdres"]
    id_lowres = tag_registry["meta/lowres"]
    id_tall = tag_registry["meta/tall_image"]
    id_safe = tag_registry["rating/safe"]
    id_marker = tag_registry["meta/booru_tagged"]

    assert write_events == [
        # Micro-batch 1: upsert all unknown tags including PROCESSED_MARKER
        (
            "PUT",
            "/api/tags",
            {
                "tags": [
                    "general/1girl",
                    "meta/absurdres",
                    "meta/lowres",
                    "meta/tall_image",
                    "meta/booru_tagged",
                ]
            },
        ),
        ("PUT", f"/api/tags/{id_1girl}/assets", {"ids": ["img-1", "img-2"]}),
        ("PUT", f"/api/tags/{id_absurdres}/assets", {"ids": ["img-1"]}),
        ("PUT", f"/api/tags/{id_lowres}/assets", {"ids": ["img-2"]}),
        ("PUT", f"/api/tags/{id_tall}/assets", {"ids": ["img-2"]}),
        ("PUT", f"/api/tags/{id_marker}/assets", {"ids": ["img-1", "img-2"]}),
        # Micro-batch 2: only unknown rating/safe is upserted; 1girl & marker hit cache
        ("PUT", "/api/tags", {"tags": ["rating/safe"]}),
        ("PUT", f"/api/tags/{id_1girl}/assets", {"ids": ["img-3"]}),
        ("PUT", f"/api/tags/{id_safe}/assets", {"ids": ["img-3"]}),
        ("PUT", f"/api/tags/{id_marker}/assets", {"ids": ["img-3"]}),
    ]


def test_batch_runner_force_mode_unbinds_only_managed_old_tags_and_preserves_personal_tags() -> None:
    events: list[tuple[str, str, object]] = []
    next_id = 100

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/album-force":
            return _json_response(
                200,
                {
                    "id": "album-force",
                    "albumName": "Force Album",
                    "assets": [
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "retag_1.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [
                                {"id": "old-marker", "value": "meta/booru_tagged"},
                                {"id": "old-gen", "value": "general/old_tag"},
                                {"id": "", "value": "artist/legacy_author"},
                                {"id": "user-1", "value": "vacation"},
                                {"id": "user-2", "value": "精选"},
                            ],
                        },
                        {
                            "id": "a-2",
                            "type": "IMAGE",
                            "originalFileName": "retag_2.png",
                            "exifInfo": {"exifImageWidth": 1600, "exifImageHeight": 1200},
                            "tags": [
                                {"id": "old-gen", "value": "general/old_tag"},
                                {"id": "old-rating", "value": "rating/explicit"},
                                {"id": "user-3", "value": "favorites/2026"},
                            ],
                        },
                    ],
                },
            )

        body = json.loads(request.content.decode("utf-8")) if request.content else None
        events.append((request.method, request.url.path, body))

        if request.method == "PUT" and request.url.path == "/api/tags":
            assert isinstance(body, dict)
            created = []
            nonlocal next_id
            for value in body["tags"]:
                if value == "artist/legacy_author":
                    tid = "resolved-artist"
                elif value == "meta/booru_tagged":
                    tid = "new-marker"
                else:
                    next_id += 1
                    tid = f"new-{next_id}"
                created.append({"id": tid, "value": value})
            return _json_response(200, created)

        if request.method in {"DELETE", "PUT"} and request.url.path.endswith("/assets"):
            return httpx.Response(status_code=204)

        if request.method == "GET" and request.url.path.endswith("/thumbnail"):
            return httpx.Response(
                status_code=200,
                headers={"content-type": "image/png"},
                content=_png_bytes(),
            )

        return _json_response(404, {"message": "Not found"})

    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        batch=BatchConfig(size=10),
        albums=[AlbumConfig(id="album-force", mode="anime")],
    )

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(config=config, client=client)
        summary = runner.run(dry_run=False, force=True)

    assert summary.processed_count == 2

    delete_calls = [e for e in events if e[0] == "DELETE"]
    put_calls = [e for e in events if e[0] == "PUT"]

    # Must detach managed tags only; personal tags (user-1, user-2, user-3) must NEVER be touched
    assert delete_calls == [
        ("DELETE", "/api/tags/old-marker/assets", {"ids": ["a-1"]}),
        ("DELETE", "/api/tags/old-gen/assets", {"ids": ["a-1", "a-2"]}),
        ("DELETE", "/api/tags/resolved-artist/assets", {"ids": ["a-1"]}),
        ("DELETE", "/api/tags/old-rating/assets", {"ids": ["a-2"]}),
    ]
    deleted_paths = {path for _, path, _ in delete_calls}
    assert "/api/tags/user-1/assets" not in deleted_paths
    assert "/api/tags/user-2/assets" not in deleted_paths
    assert "/api/tags/user-3/assets" not in deleted_paths

    # All DELETEs must happen before attaching new tags and final PROCESSED_MARKER
    last_delete_idx = max(i for i, e in enumerate(events) if e[0] == "DELETE")
    first_attach_idx = min(
        i for i, e in enumerate(events) if e[0] == "PUT" and e[1].endswith("/assets")
    )
    assert last_delete_idx < first_attach_idx
    assert put_calls[-1] == (
        "PUT",
        "/api/tags/new-marker/assets",
        {"ids": ["a-1", "a-2"]},
    )


def test_batch_runner_fast_fails_when_micro_batch_write_back_fails() -> None:
    write_requests: list[tuple[str, str]] = []
    base_transport = _sample_transport([])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/api/tags":
            write_requests.append((request.method, request.url.path))
            body = json.loads(request.content.decode("utf-8"))
            return _json_response(
                200, [{"id": f"id-{i}", "value": v} for i, v in enumerate(body["tags"])]
            )
        if request.method == "PUT" and request.url.path.endswith("/assets"):
            write_requests.append((request.method, request.url.path))
            return _json_response(500, {"message": "Internal server error"})
        return base_transport.handle_request(request)

    config = _sample_config(batch_size=2)

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(config=config, client=client)
        with pytest.raises(ImmichApiError) as exc_info:
            runner.run(dry_run=False)

    assert exc_info.value.status_code == 500
    # Batch 1 aborted at first asset-binding call; batch 2 never ran
    assert len(write_requests) == 2


def test_batch_runner_fetches_configured_thumbnail_size_decodes_in_memory_and_closes_image() -> None:
    seen_images: list[Image.Image] = []
    seen_thumb_requests: list[tuple[str, str]] = []

    class RecordingVisionPipeline:
        def predict(
            self, image: Image.Image, mode: RoutingMode
        ) -> Sequence[tuple[str, str]]:
            # Prior image must already be closed before processing the next asset
            for prev in seen_images:
                with pytest.raises(ValueError, match="closed image"):
                    prev.getpixel((0, 0))
            assert image.getpixel((0, 0)) == (10, 20, 30)
            seen_images.append(image)
            return [("general", "1girl")]

    base_transport = _sample_transport([])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/thumbnail"):
            seen_thumb_requests.append((request.url.path, request.url.params.get("size", "")))
        return base_transport.handle_request(request)

    config = AppConfig(
        immich=ImmichConfig(
            base_url="http://immich.local:2283",
            api_key="secret",
            thumbnail_size="thumbnail",
        ),
        batch=BatchConfig(size=2),
        albums=[AlbumConfig(name="Anime Album", mode="anime")],
    )

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(
            config=config,
            client=client,
            vision_pipeline=RecordingVisionPipeline(),
        )
        summary = runner.run(dry_run=True)

    assert summary.processed_count == 2
    assert seen_thumb_requests == [
        ("/api/assets/img-1/thumbnail", "thumbnail"),
        ("/api/assets/img-2/thumbnail", "thumbnail"),
    ]
    assert len(seen_images) == 2
    for img in seen_images:
        with pytest.raises(ValueError, match="closed image"):
            img.getpixel((0, 0))


def test_batch_runner_skips_missing_or_corrupt_thumbnails_with_warning_and_writes_remaining(
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_events: list[tuple[str, str, object]] = []
    predicted_count = 0

    class CountingVisionPipeline:
        def predict(
            self, image: Image.Image, mode: RoutingMode
        ) -> Sequence[tuple[str, str]]:
            nonlocal predicted_count
            predicted_count += 1
            return [("general", "1girl")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/album-mixed":
            return _json_response(
                200,
                {
                    "id": "album-mixed",
                    "albumName": "Mixed Health",
                    "assets": [
                        {
                            "id": "img-404",
                            "type": "IMAGE",
                            "originalFileName": "missing_thumb.png",
                            "exifInfo": {"exifImageWidth": 1600, "exifImageHeight": 1200},
                            "tags": [],
                        },
                        {
                            "id": "img-corrupt",
                            "type": "IMAGE",
                            "originalFileName": "corrupt_bytes.png",
                            "exifInfo": {"exifImageWidth": 1600, "exifImageHeight": 1200},
                            "tags": [],
                        },
                        {
                            "id": "img-ok",
                            "type": "IMAGE",
                            "originalFileName": "healthy.png",
                            "exifInfo": {"exifImageWidth": 3200, "exifImageHeight": 2400},
                            "tags": [],
                        },
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/api/assets/img-404/thumbnail":
            return _json_response(404, {"message": "Thumbnail not generated"})
        if request.method == "GET" and request.url.path == "/api/assets/img-corrupt/thumbnail":
            return httpx.Response(
                status_code=200,
                headers={"content-type": "image/png"},
                content=b"not-a-valid-image-stream",
            )
        if request.method == "GET" and request.url.path == "/api/assets/img-ok/thumbnail":
            return httpx.Response(
                status_code=200,
                headers={"content-type": "image/png"},
                content=_png_bytes(),
            )
        if request.method == "PUT" and request.url.path == "/api/tags":
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            return _json_response(
                200, [{"id": f"t-{i}", "value": v} for i, v in enumerate(body["tags"])]
            )
        if request.method == "PUT" and request.url.path.endswith("/assets"):
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            return httpx.Response(status_code=204)
        return _json_response(404, {"message": "Not found"})

    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        batch=BatchConfig(size=10),
        albums=[AlbumConfig(id="album-mixed", mode="anime")],
    )

    caplog.set_level(logging.WARNING)
    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(
            config=config, client=client, vision_pipeline=CountingVisionPipeline()
        )
        summary = runner.run(dry_run=False)

    assert summary.total_eligible == 3
    assert summary.processed_count == 1
    assert summary.skipped_count == 2
    assert predicted_count == 1
    assert [r.asset_id for r in summary.results] == ["img-ok"]

    warning_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("img-404" in msg and "missing_thumb.png" in msg for msg in warning_messages)
    assert any("img-corrupt" in msg and "corrupt_bytes.png" in msg for msg in warning_messages)

    # Only img-ok receives tags and meta/booru_tagged
    assert write_events == [
        ("PUT", "/api/tags", {"tags": ["general/1girl", "meta/absurdres", "meta/booru_tagged"]}),
        ("PUT", "/api/tags/t-0/assets", {"ids": ["img-ok"]}),
        ("PUT", "/api/tags/t-1/assets", {"ids": ["img-ok"]}),
        ("PUT", "/api/tags/t-2/assets", {"ids": ["img-ok"]}),
    ]


def test_batch_runner_skips_decompression_bomb_thumbnail_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/album-bomb":
            return _json_response(
                200,
                {
                    "id": "album-bomb",
                    "albumName": "Bomb Check",
                    "assets": [
                        {
                            "id": "img-bomb",
                            "type": "IMAGE",
                            "originalFileName": "huge.png",
                            "exifInfo": None,
                            "tags": [],
                        },
                        {
                            "id": "img-small",
                            "type": "IMAGE",
                            "originalFileName": "ok.png",
                            "exifInfo": None,
                            "tags": [],
                        },
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/api/assets/img-bomb/thumbnail":
            return httpx.Response(200, content=_png_bytes(4, 4))  # 16 px > 4 px limit
        if request.method == "GET" and request.url.path == "/api/assets/img-small/thumbnail":
            return httpx.Response(200, content=_png_bytes(2, 2))  # 4 px <= 4 px limit
        return _json_response(404, {})

    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        albums=[AlbumConfig(id="album-bomb", mode="anime")],
    )

    caplog.set_level(logging.WARNING)
    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(config=config, client=client)
        summary = runner.run(dry_run=True)

    assert summary.processed_count == 1
    assert summary.skipped_count == 1
    assert [r.asset_id for r in summary.results] == ["img-small"]
    assert any(
        "img-bomb" in rec.getMessage() and "huge.png" in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    )


@pytest.mark.parametrize("status_code", [401, 403])
def test_batch_runner_fast_fails_on_thumbnail_auth_error(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/album-auth":
            return _json_response(
                200,
                {
                    "id": "album-auth",
                    "albumName": "Auth Check",
                    "assets": [
                        {
                            "id": "img-1",
                            "type": "IMAGE",
                            "originalFileName": "pic.png",
                            "exifInfo": None,
                            "tags": [],
                        }
                    ],
                },
            )
        if request.method == "GET" and request.url.path.endswith("/thumbnail"):
            return _json_response(status_code, {"message": "Unauthorized"})
        return _json_response(404, {})

    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        albums=[AlbumConfig(id="album-auth", mode="anime")],
    )

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(config=config, client=client)
        with pytest.raises(ImmichApiError) as exc_info:
            runner.run(dry_run=False)

    assert exc_info.value.status_code == status_code


def test_anime_album_end_to_end_with_default_vision_pipeline_and_truncated_image_skip() -> None:
    seen_thresholds: list[tuple[float, float]] = []
    write_events: list[tuple[str, str, object]] = []
    tag_ids: dict[str, str] = {}

    class FakeAnimeBackend:
        def extract_pixai(
            self,
            image: Image.Image,
            *,
            general_threshold: float,
            character_threshold: float,
        ) -> tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]:
            seen_thresholds.append((general_threshold, character_threshold))
            if image.size == (3, 3):
                return (
                    {"1girl": 0.96, "Blue Eyes": 0.8},
                    {"Artoria Pendragon": 0.92},
                    {"Fate/Grand Order": 0.9},
                )
            return ({"solo": 0.9}, {}, {})

        def extract_rating_scores(self, image: Image.Image) -> Mapping[str, float]:
            if image.size == (3, 3):
                return {"general": 0.1, "sensitive": 0.2, "questionable": 0.65, "explicit": 0.05}
            return {"general": 0.9, "sensitive": 0.05, "questionable": 0.03, "explicit": 0.02}

        def classify_medium(self, image: Image.Image) -> tuple[str, float]:
            return ("illustration", 0.95) if image.size == (3, 3) else ("not_painting", 0.9)

        def classify_completeness(self, image: Image.Image) -> tuple[str, float]:
            return ("polished", 0.94) if image.size == (3, 3) else ("monochrome", 0.88)

    valid_3x3 = _png_bytes(3, 3)
    valid_2x2 = _png_bytes(2, 2)
    truncated_png = valid_2x2[:16]  # valid PNG header but corrupted/truncated body on load()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/albums/album-anime-e2e":
            return _json_response(
                200,
                {
                    "id": "album-anime-e2e",
                    "albumName": "Anime E2E",
                    "assets": [
                        {
                            "id": "a-illu",
                            "type": "IMAGE",
                            "originalFileName": "saber.png",
                            "exifInfo": {"exifImageWidth": 4000, "exifImageHeight": 3000},
                            "tags": [],
                        },
                        {
                            "id": "a-not-painting",
                            "type": "IMAGE",
                            "originalFileName": "ui_screenshot.png",
                            "exifInfo": {"exifImageWidth": 1000, "exifImageHeight": 800},
                            "tags": [],
                        },
                        {
                            "id": "a-truncated",
                            "type": "IMAGE",
                            "originalFileName": "truncated.png",
                            "exifInfo": {"exifImageWidth": 1000, "exifImageHeight": 800},
                            "tags": [],
                        },
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/api/assets/a-illu/thumbnail":
            return httpx.Response(200, content=valid_3x3)
        if request.method == "GET" and request.url.path == "/api/assets/a-not-painting/thumbnail":
            return httpx.Response(200, content=valid_2x2)
        if request.method == "GET" and request.url.path == "/api/assets/a-truncated/thumbnail":
            return httpx.Response(200, content=truncated_png)
        if request.method == "PUT" and request.url.path == "/api/tags":
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            created = []
            for value in body["tags"]:
                tid = tag_ids.setdefault(value, f"tid-{len(tag_ids) + 1}")
                created.append({"id": tid, "value": value})
            return _json_response(200, created)
        if request.method == "PUT" and request.url.path.endswith("/assets"):
            body = json.loads(request.content.decode("utf-8"))
            write_events.append((request.method, request.url.path, body))
            return httpx.Response(status_code=204)
        return _json_response(404, {})

    config = AppConfig(
        immich=ImmichConfig(base_url="http://immich.local:2283", api_key="secret"),
        batch=BatchConfig(size=10),
        pipeline=PipelinesConfig(
            anime=PipelineThresholdConfig(general_threshold=0.42, character_threshold=0.88)
        ),
        albums=[AlbumConfig(id="album-anime-e2e", mode="anime")],
    )
    vision_pipeline = DefaultVisionPipeline(
        config.pipeline,
        anime_backend=FakeAnimeBackend(),
    )

    with ImmichClient.from_config(config, transport=httpx.MockTransport(handler)) as client:
        runner = BatchRunner(
            config=config, client=client, vision_pipeline=vision_pipeline
        )
        summary = runner.run(dry_run=False)

    assert summary.processed_count == 2
    assert summary.skipped_count == 1
    assert seen_thresholds == [(0.42, 0.88), (0.42, 0.88)]

    by_id = {r.asset_id: r for r in summary.results}
    assert by_id["a-illu"].tags == [
        "general/1girl",
        "general/blue_eyes",
        "character/artoria_pendragon",
        "copyright/fate_grand_order",
        "rating/questionable",
        "meta/illustration",
        "meta/polished",
        "meta/absurdres",
    ]
    assert by_id["a-not-painting"].tags == [
        "general/solo",
        "rating/general",
        "meta/monochrome",
    ]
    assert "meta/not_painting" not in by_id["a-not-painting"].tags
    assert "a-truncated" not in by_id

    marker_id = tag_ids["meta/booru_tagged"]
    assert write_events[-1] == (
        "PUT",
        f"/api/tags/{marker_id}/assets",
        {"ids": ["a-illu", "a-not-painting"]},
    )
