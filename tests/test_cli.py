"""End-to-end CLI tests driving cli.main with real TOML files and httpx.MockTransport."""

import json
from pathlib import Path

import httpx
import pytest

from immich_booru_tagger.cli import main


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
        return _json_response(404, {"message": "Not found"})

    exit_code = main(
        ["--config", str(cfg_file), "--dry-run", "--limit", "1"],
        transport=httpx.MockTransport(handler),
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
        return _json_response(404, {})

    exit_code = main(
        ["-c", str(cfg_file), "--dry-run", "--force"],
        transport=httpx.MockTransport(handler),
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
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "dry_run=False" in out
    assert "[dry-run]" not in out
    assert write_events == [
        ("DELETE", "/api/tags/old-m/assets", {"ids": ["a-1"]}),
        ("PUT", "/api/tags", {"tags": ["meta/absurdres", "meta/booru_tagged"]}),
        ("PUT", "/api/tags/new-0/assets", {"ids": ["a-1"]}),
        ("PUT", "/api/tags/new-1/assets", {"ids": ["a-1"]}),
    ]
