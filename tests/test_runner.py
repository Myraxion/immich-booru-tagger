"""End-to-end tests for BatchRunner through ImmichClient + httpx.MockTransport."""

import json
from collections.abc import Sequence

import httpx
import pytest

from immich_booru_tagger.config import AlbumConfig, AppConfig, BatchConfig, ImmichConfig, RoutingMode
from immich_booru_tagger.immich import ImmichAsset, ImmichClient
from immich_booru_tagger.runner import BatchRunner


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
            self, asset: ImmichAsset, mode: RoutingMode
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


def test_batch_runner_rejects_non_dry_run_until_write_back_is_implemented() -> None:
    requests_log: list[httpx.Request] = []
    config = _sample_config()

    with ImmichClient.from_config(
        config, transport=_sample_transport(requests_log)
    ) as client:
        runner = BatchRunner(config=config, client=client)

        with pytest.raises(NotImplementedError):
            runner.run(dry_run=False)
