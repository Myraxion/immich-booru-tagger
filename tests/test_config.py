"""Tests for TOML configuration loading, environment overrides, and validation."""

from pathlib import Path

import pytest

from immich_booru_tagger.config import ConfigError, load_config


def test_load_config_parses_defaults_and_normalizes_base_url(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283///"
api_key = "secret-key"

[[albums]]
name = "Anime Art"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_file, env={})

    assert config.immich.base_url == "http://127.0.0.1:2283"
    assert config.immich.api_key == "secret-key"
    assert config.immich.thumbnail_size == "preview"
    assert config.immich.timeout == 30.0
    assert config.batch.size == 20
    assert config.pipeline.anime.general_threshold == 0.35
    assert config.pipeline.anime.character_threshold == 0.85
    assert config.pipeline.real.general_threshold == 0.35
    assert config.pipeline.real.character_threshold == 0.85
    assert len(config.albums) == 1
    assert config.albums[0].name == "Anime Art"
    assert config.albums[0].id is None
    assert config.albums[0].mode == "auto"


def test_load_config_parses_explicit_values(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "https://immich.example.com"
api_key = "file-key"
thumbnail_size = "thumbnail"
timeout = 12.5

[batch]
size = 10

[pipeline.anime]
general_threshold = 0.4
character_threshold = 0.9

[pipeline.real]
general_threshold = 0.25
character_threshold = 0.75

[[albums]]
id = "11111111-1111-1111-1111-111111111111"
mode = "real"

[[albums]]
name = "Mixed"
mode = "anime"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_file, env={})

    assert config.immich.thumbnail_size == "thumbnail"
    assert config.immich.timeout == 12.5
    assert config.batch.size == 10
    assert config.pipeline.anime.general_threshold == 0.4
    assert config.pipeline.anime.character_threshold == 0.9
    assert config.pipeline.real.general_threshold == 0.25
    assert config.pipeline.real.character_threshold == 0.75
    assert config.albums[0].id == "11111111-1111-1111-1111-111111111111"
    assert config.albums[0].mode == "real"
    assert config.albums[1].name == "Mixed"
    assert config.albums[1].mode == "anime"


def test_load_config_allows_env_vars_to_override_and_supply_credentials(
    tmp_path: Path,
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://old.local:2283"

[[albums]]
name = "Art"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        cfg_file,
        env={
            "IMMICH_BASE_URL": "https://photos.example.com/",
            "IMMICH_API_KEY": "env-secret-key",
        },
    )

    assert config.immich.base_url == "https://photos.example.com"
    assert config.immich.api_key == "env-secret-key"


def test_load_config_ignores_whitespace_only_env_vars(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "file-key"

[[albums]]
name = "Art"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        cfg_file,
        env={"IMMICH_BASE_URL": "   ", "IMMICH_API_KEY": ""},
    )

    assert config.immich.base_url == "http://127.0.0.1:2283"
    assert config.immich.api_key == "file-key"


def test_load_config_fails_when_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml", env={})


@pytest.mark.parametrize(
    "toml_body",
    [
        # Missing albums
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
""",
        # Album missing both id and name
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[[albums]]
mode = "auto"
""",
        # Album specifying both id and name
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[[albums]]
id = "abc"
name = "Art"
""",
        # Album specifying empty id alongside name
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[[albums]]
id = ""
name = "Art"
""",
        # Invalid mode
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[[albums]]
name = "Art"
mode = "invalid"
""",
        # Invalid thumbnail_size
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
thumbnail_size = "original"
[[albums]]
name = "Art"
""",
        # Non-positive batch size
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[batch]
size = 0
[[albums]]
name = "Art"
""",
        # Threshold out of range [0.0, 1.0]
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[pipeline.anime]
general_threshold = 1.1
[[albums]]
name = "Art"
""",
        # Negative threshold
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "k"
[pipeline.real]
character_threshold = -0.1
[[albums]]
name = "Art"
""",
        # Missing API key (not in TOML nor env)
        """
[immich]
base_url = "http://127.0.0.1:2283"
[[albums]]
name = "Art"
""",
        # Invalid base_url scheme
        """
[immich]
base_url = "ftp://127.0.0.1:2283"
api_key = "k"
[[albums]]
name = "Art"
""",
    ],
)
def test_load_config_fast_fails_on_invalid_schema(
    tmp_path: Path, toml_body: str
) -> None:
    cfg_file = tmp_path / "invalid.toml"
    cfg_file.write_text(toml_body.strip(), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(cfg_file, env={})


def test_load_config_parses_album_artist_patterns(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "secret-key"

[[albums]]
name = "Pixiv"
artist_patterns = ['^\\[(?P<artist>[^\\]]+)\\]', '^pixiv_\\d+_(.+)$']

[[albums]]
name = "Default"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_file, env={})

    assert config.albums[0].artist_patterns == [
        r"^\[(?P<artist>[^\]]+)\]",
        r"^pixiv_\d+_(.+)$",
    ]
    assert config.albums[1].artist_patterns == []


@pytest.mark.parametrize(
    "pattern",
    [
        "[unclosed",
        r"^pixiv_\d+_.*$",
    ],
)
def test_load_config_rejects_invalid_or_groupless_artist_patterns(
    tmp_path: Path, pattern: str
) -> None:
    cfg_file = tmp_path / "invalid_pattern.toml"
    cfg_file.write_text(
        f"""
[immich]
base_url = "http://127.0.0.1:2283"
api_key = "secret-key"

[[albums]]
name = "Pixiv"
artist_patterns = [{pattern!r}]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(cfg_file, env={})
