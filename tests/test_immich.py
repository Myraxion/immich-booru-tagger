"""Tests for ImmichClient album resolution and asset parsing via httpx.MockTransport."""

import json

import httpx
import pytest

from immich_booru_tagger.config import AlbumConfig
from immich_booru_tagger.immich import (
    AlbumNotFoundError,
    AmbiguousAlbumError,
    ImmichApiError,
    ImmichClient,
)


def _json_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def test_resolve_target_album_by_id_parses_assets_and_sends_api_key() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.method == "GET" and request.url.path == "/api/albums/album-1":
            return _json_response(
                200,
                {
                    "id": "album-1",
                    "albumName": "Illustrations",
                    "assets": [
                        {
                            "id": "asset-1",
                            "type": "IMAGE",
                            "originalFileName": "art_01.png",
                            "exifInfo": {
                                "exifImageWidth": 3840,
                                "exifImageHeight": 2160,
                            },
                            "tags": [
                                {"id": "tag-1", "value": "meta/booru_tagged"},
                                {"id": "tag-2", "name": "favorites/top"},
                            ],
                        },
                        {
                            "id": "asset-2",
                            "type": "VIDEO",
                            "originalFileName": "clip.mp4",
                            "exifInfo": None,
                            "tags": None,
                        },
                    ],
                },
            )
        return _json_response(404, {"message": "Not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="my-secret-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        albums = client.resolve_target_albums([AlbumConfig(id="album-1", mode="anime")])

    assert seen_requests
    assert all(req.headers["x-api-key"] == "my-secret-key" for req in seen_requests)
    assert all(req.url.host == "immich.local" for req in seen_requests)

    assert len(albums) == 1
    album = albums[0]
    assert album.id == "album-1"
    assert album.name == "Illustrations"
    assert album.mode == "anime"
    assert len(album.assets) == 2

    first = album.assets[0]
    assert first.id == "asset-1"
    assert first.type == "IMAGE"
    assert first.original_file_name == "art_01.png"
    assert first.width == 3840
    assert first.height == 2160
    assert first.tag_values == {"meta/booru_tagged", "favorites/top"}

    second = album.assets[1]
    assert second.type == "VIDEO"
    assert second.width is None
    assert second.height is None
    assert second.tag_values == set()


def test_resolve_target_album_by_name_looks_up_and_fetches_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/albums":
            return _json_response(
                200,
                [
                    {"id": "uuid-other", "albumName": "Other"},
                    {"id": "uuid-target", "albumName": "Cosplay"},
                ],
            )
        if request.url.path == "/api/albums/uuid-target":
            return _json_response(
                200,
                {
                    "id": "uuid-target",
                    "albumName": "Cosplay",
                    "assets": [
                        {
                            "id": "a-1",
                            "type": "IMAGE",
                            "originalFileName": "photo.jpg",
                            "exifInfo": {"exifImageWidth": 1600, "exifImageHeight": 1200},
                            "tags": [],
                        }
                    ],
                },
            )
        return _json_response(404, {"message": "Not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler),
    ) as client:
        albums = client.resolve_target_albums([AlbumConfig(name="Cosplay", mode="real")])

    assert len(albums) == 1
    assert albums[0].id == "uuid-target"
    assert albums[0].name == "Cosplay"
    assert albums[0].mode == "real"
    assert [asset.id for asset in albums[0].assets] == ["a-1"]


def test_resolve_target_album_by_name_fails_when_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, [{"id": "uuid-1", "albumName": "Other"}])

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AlbumNotFoundError, match="Missing Album"):
            client.resolve_target_albums([AlbumConfig(name="Missing Album")])


def test_resolve_target_album_by_name_fails_and_lists_uuids_on_ambiguous_match() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            [
                {"id": "11111111-aaaa-bbbb-cccc-000000000001", "albumName": "Duplicate"},
                {"id": "22222222-aaaa-bbbb-cccc-000000000002", "albumName": "Duplicate"},
                {"id": "33333333-aaaa-bbbb-cccc-000000000003", "albumName": "Unique"},
            ],
        )

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AmbiguousAlbumError) as exc_info:
            client.resolve_target_albums([AlbumConfig(name="Duplicate")])

    message = str(exc_info.value)
    assert "Duplicate" in message
    assert "11111111-aaaa-bbbb-cccc-000000000001" in message
    assert "22222222-aaaa-bbbb-cccc-000000000002" in message


def test_resolve_target_album_by_id_fails_on_404() -> None:
    def handler_404(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"message": "Album not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler_404),
    ) as client:
        with pytest.raises(AlbumNotFoundError):
            client.resolve_target_albums([AlbumConfig(id="nonexistent-id")])


def test_resolve_target_album_fails_on_auth_error() -> None:
    def handler_401(request: httpx.Request) -> httpx.Response:
        return _json_response(401, {"message": "Invalid API key"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="bad-key",
        transport=httpx.MockTransport(handler_401),
    ) as client:
        with pytest.raises(ImmichApiError):
            client.resolve_target_albums([AlbumConfig(id="album-1")])
