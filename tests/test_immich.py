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


def test_ensure_tags_caches_tag_ids_and_skips_upsert_on_cache_hit() -> None:
    upsert_calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path == "/api/tags":
            body = json.loads(request.content.decode("utf-8"))
            tags = body["tags"]
            upsert_calls.append(tags)
            return _json_response(
                200,
                [
                    {"id": f"id-{i}", "name": tag} if i % 2 else {"id": f"id-{i}", "value": tag}
                    for i, tag in enumerate(tags, start=len(upsert_calls) * 10)
                ],
            )
        return _json_response(404, {"message": "Not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler),
    ) as client:
        first = client.ensure_tags(["general/1girl", "meta/absurdres", "general/1girl"])
        second = client.ensure_tags(["meta/absurdres", "meta/booru_tagged"])
        third = client.ensure_tags(["general/1girl", "meta/booru_tagged"])

    assert upsert_calls == [
        ["general/1girl", "meta/absurdres"],
        ["meta/booru_tagged"],
    ]
    assert first == {"general/1girl": "id-10", "meta/absurdres": "id-11"}
    assert second == {"meta/absurdres": "id-11", "meta/booru_tagged": "id-20"}
    assert third == {"general/1girl": "id-10", "meta/booru_tagged": "id-20"}


def test_tag_and_untag_assets_send_expected_requests_and_handle_empty_body() -> None:
    seen: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "PUT" and request.url.path == "/api/tags/tag-1/assets":
            return httpx.Response(status_code=204)
        if request.method == "DELETE" and request.url.path == "/api/tags/tag-2/assets":
            return _json_response(200, [{"id": "a-1", "success": True}])
        return _json_response(404, {"message": "Not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.tag_assets("tag-1", ["a-1", "a-2"])
        client.untag_assets("tag-2", ["a-1"])

    assert seen == [
        ("PUT", "/api/tags/tag-1/assets", {"ids": ["a-1", "a-2"]}),
        ("DELETE", "/api/tags/tag-2/assets", {"ids": ["a-1"]}),
    ]


def test_fetch_asset_thumbnail_requests_size_and_returns_raw_bytes() -> None:
    seen_requests: list[httpx.Request] = []
    png_payload = b"\x89PNG\r\n\x1a\nfake-thumbnail-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.method == "GET" and request.url.path == "/api/assets/asset-99/thumbnail":
            return httpx.Response(
                status_code=200,
                headers={"content-type": "image/png"},
                content=png_payload,
            )
        return _json_response(404, {"message": "Not found"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="my-secret-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        data = client.fetch_asset_thumbnail("asset-99", size="thumbnail")

    assert data == png_payload
    assert len(seen_requests) == 1
    assert seen_requests[0].url.path == "/api/assets/asset-99/thumbnail"
    assert seen_requests[0].url.params["size"] == "thumbnail"
    assert seen_requests[0].headers["x-api-key"] == "my-secret-key"


def test_fetch_asset_thumbnail_raises_immich_api_error_on_http_or_transport_failure() -> None:
    def handler_404(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"message": "Thumbnail missing"})

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler_404),
    ) as client:
        with pytest.raises(ImmichApiError) as exc_info:
            client.fetch_asset_thumbnail("missing-asset")
    assert exc_info.value.status_code == 404

    def handler_transport_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with ImmichClient(
        base_url="http://immich.local:2283",
        api_key="k",
        transport=httpx.MockTransport(handler_transport_error),
    ) as client:
        with pytest.raises(ImmichApiError) as exc_info2:
            client.fetch_asset_thumbnail("asset-1")
    assert exc_info2.value.status_code is None
