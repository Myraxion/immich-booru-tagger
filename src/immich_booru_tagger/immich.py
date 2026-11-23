"""Immich REST API client and domain response models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from immich_booru_tagger.config import AlbumConfig, AppConfig, RoutingMode, ThumbnailSize
from immich_booru_tagger.rules import PROCESSED_MARKER, is_managed_tag
from immich_booru_tagger.timefilter import ensure_aware


class ImmichError(RuntimeError):
    """Base error for Immich API interactions."""


class ImmichApiError(ImmichError):
    """Raised when Immich returns an unexpected HTTP error or transport failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AlbumNotFoundError(ImmichError):
    """Raised when a configured Target Album cannot be found."""


class AmbiguousAlbumError(ImmichError):
    """Raised when an album name matches more than one album."""


class ImmichTag(BaseModel):
    """Tag attached to an Immich asset."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    value: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_name_to_value(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("value") and data.get("name"):
            return {**data, "value": data["name"]}
        return data


class ExifInfo(BaseModel):
    """EXIF metadata subset needed for deterministic dimension rules."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    exif_image_width: int | None = Field(default=None, alias="exifImageWidth")
    exif_image_height: int | None = Field(default=None, alias="exifImageHeight")


class ImmichAsset(BaseModel):
    """Immich asset metadata parsed from album endpoints."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    type: str
    original_file_name: str = Field(default="", alias="originalFileName")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    exif_info: ExifInfo | None = Field(default=None, alias="exifInfo")
    tags: list[ImmichTag] = Field(default_factory=list)
    tags_present: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _track_tags_presence(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tags" in data and data["tags"] is not None:
            return {**data, "tags_present": True}
        return data

    @field_validator("original_file_name", mode="before")
    @classmethod
    def _coerce_null_file_name(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_null_tags(cls, value: Any) -> Any:
        return [] if value is None else value

    @property
    def width(self) -> int | None:
        return self.exif_info.exif_image_width if self.exif_info is not None else None

    @property
    def height(self) -> int | None:
        return self.exif_info.exif_image_height if self.exif_info is not None else None

    @property
    def tag_values(self) -> set[str]:
        return {tag.value for tag in self.tags}


class _AlbumSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    album_name: str = Field(default="", alias="albumName")


class _AlbumDetailPayload(_AlbumSummaryPayload):
    asset_count: int = Field(default=0, alias="assetCount")
    assets: list[ImmichAsset] = Field(default_factory=list)


class TargetAlbum(BaseModel):
    """Resolved Target Album with its configured Routing Mode and assets."""

    id: str
    name: str
    mode: RoutingMode
    assets: list[ImmichAsset]


def _append_unique(mapping: dict[str, list[str]], key: str, value: str) -> None:
    bucket = mapping.setdefault(key, [])
    if value not in bucket:
        bucket.append(value)


def _to_utc_iso(dt: datetime) -> str:
    return ensure_aware(dt).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _within_created_at_range(
    asset: ImmichAsset,
    created_after: datetime | None,
    created_before: datetime | None,
) -> bool:
    if created_after is None and created_before is None:
        return True
    if asset.created_at is None:
        return False
    asset_ts = ensure_aware(asset.created_at, default_tz=timezone.utc)
    if created_after is not None and asset_ts < ensure_aware(created_after):
        return False
    if created_before is not None and asset_ts > ensure_aware(created_before):
        return False
    return True


class ImmichClient:
    """Synchronous HTTP client wrapper for Immich REST APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._tag_cache: dict[str, str] = {}
        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"x-api-key": api_key},
                timeout=timeout,
                transport=transport,
            )
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> ImmichClient:
        """Construct an ImmichClient from a validated AppConfig."""
        return cls(
            base_url=config.immich.base_url,
            api_key=config.immich.api_key,
            timeout=config.immich.timeout,
            transport=transport,
        )

    def __enter__(self) -> ImmichClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying httpx.Client if owned by this instance."""
        if self._owns_client:
            self._client.close()

    def resolve_target_albums(
        self,
        album_configs: Sequence[AlbumConfig],
        *,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[TargetAlbum]:
        """Resolve all configured Target Albums into concrete TargetAlbum objects."""
        all_albums: list[_AlbumSummaryPayload] | None = None
        resolved: list[TargetAlbum] = []

        for album_cfg in album_configs:
            if album_cfg.id is not None:
                album_id = album_cfg.id
            else:
                assert album_cfg.name is not None
                if all_albums is None:
                    all_albums = self._list_albums()
                album_id = self._match_album_by_name(all_albums, album_cfg.name)

            detail = self._get_album_detail(album_id)
            raw_assets = (
                detail.assets
                if detail.assets or detail.asset_count == 0
                else self._search_album_assets(
                    album_id,
                    created_after=created_after,
                    created_before=created_before,
                )
            )
            filtered_assets = [
                asset
                for asset in raw_assets
                if _within_created_at_range(asset, created_after, created_before)
            ]
            enriched_assets = [
                self._enrich_asset_tags_if_missing(asset) for asset in filtered_assets
            ]
            resolved.append(
                TargetAlbum(
                    id=detail.id,
                    name=detail.album_name,
                    mode=album_cfg.mode,
                    assets=enriched_assets,
                )
            )

        return resolved

    def _search_album_assets(
        self,
        album_id: str,
        *,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[ImmichAsset]:
        assets: list[ImmichAsset] = []
        page = 1
        while True:
            search_body: dict[str, Any] = {
                "albumIds": [album_id],
                "page": page,
                "size": 250,
                "withExif": True,
            }
            if created_after is not None:
                search_body["createdAfter"] = _to_utc_iso(created_after)
            if created_before is not None:
                search_body["createdBefore"] = _to_utc_iso(created_before)

            payload = self._request_json(
                "POST",
                "/api/search/metadata",
                json=search_body,
            )
            if not isinstance(payload, dict):
                raise ImmichApiError("Expected JSON object from POST /api/search/metadata")
            assets_section = payload.get("assets")
            if not isinstance(assets_section, dict):
                raise ImmichApiError("Missing 'assets' object in POST /api/search/metadata response")
            items = assets_section.get("items") or []
            if not isinstance(items, list):
                raise ImmichApiError("Expected 'assets.items' list from POST /api/search/metadata")
            assets.extend(ImmichAsset.model_validate(item) for item in items)

            next_page = assets_section.get("nextPage")
            if not next_page:
                break
            page = int(next_page)

        return assets

    def _enrich_asset_tags_if_missing(self, asset: ImmichAsset) -> ImmichAsset:
        if asset.type != "IMAGE" or asset.tags_present:
            return asset
        payload = self._request_json("GET", f"/api/assets/{asset.id}")
        if not isinstance(payload, dict):
            raise ImmichApiError(
                f"Expected JSON object from GET /api/assets/{asset.id}"
            )
        detailed = ImmichAsset.model_validate(payload)
        return asset.model_copy(
            update={
                "tags": detailed.tags,
                "tags_present": True,
                "exif_info": asset.exif_info or detailed.exif_info,
            }
        )

    def fetch_asset_thumbnail(
        self, asset_id: str, size: ThumbnailSize = "preview"
    ) -> bytes:
        """Fetch raw thumbnail bytes for a single Immich asset."""
        response = self._send_request(
            "GET", f"/api/assets/{asset_id}/thumbnail", params={"size": size}
        )
        return response.content

    def ensure_tags(self, tag_values: Sequence[str]) -> dict[str, str]:
        """Upsert any unknown tags in Immich and return value -> id mapping."""
        unique_values = list(dict.fromkeys(tag_values))
        unknown = [value for value in unique_values if value not in self._tag_cache]
        if unknown:
            payload = self._request_json("PUT", "/api/tags", json={"tags": unknown})
            if not isinstance(payload, list):
                raise ImmichApiError("Expected JSON list from PUT /api/tags")
            for item in payload:
                parsed = ImmichTag.model_validate(item)
                if parsed.id:
                    self._tag_cache[parsed.value] = parsed.id

            missing = [value for value in unknown if value not in self._tag_cache]
            if missing:
                raise ImmichApiError(
                    f"PUT /api/tags did not return tag IDs for: {', '.join(missing)}"
                )

        return {value: self._tag_cache[value] for value in unique_values}

    def tag_assets(self, tag_id: str, asset_ids: Sequence[str]) -> None:
        """Attach an existing tag ID to the given asset IDs."""
        if not asset_ids:
            return
        self._request_json("PUT", f"/api/tags/{tag_id}/assets", json={"ids": list(asset_ids)})

    def untag_assets(self, tag_id: str, asset_ids: Sequence[str]) -> None:
        """Detach an existing tag ID from the given asset IDs."""
        if not asset_ids:
            return
        self._request_json("DELETE", f"/api/tags/{tag_id}/assets", json={"ids": list(asset_ids)})

    def sync_batch_tags(
        self,
        batch_updates: Sequence[tuple[ImmichAsset, Sequence[str]]],
        *,
        force: bool = False,
    ) -> None:
        """Persist one Micro-batch of asset tags and PROCESSED_MARKER to Immich."""
        if not batch_updates:
            return

        if force:
            self._unbind_managed_tags([asset for asset, _ in batch_updates])

        assets_by_tag: dict[str, list[str]] = {}
        all_asset_ids: list[str] = []
        for asset, tags in batch_updates:
            if asset.id not in all_asset_ids:
                all_asset_ids.append(asset.id)
            for tag_value in dict.fromkeys(tags):
                if tag_value == PROCESSED_MARKER:
                    continue
                _append_unique(assets_by_tag, tag_value, asset.id)

        tags_to_ensure = [*assets_by_tag.keys(), PROCESSED_MARKER]
        tag_id_map = self.ensure_tags(tags_to_ensure)

        for tag_value, asset_ids in assets_by_tag.items():
            self.tag_assets(tag_id_map[tag_value], asset_ids)

        self.tag_assets(tag_id_map[PROCESSED_MARKER], all_asset_ids)

    def _unbind_managed_tags(self, assets: Sequence[ImmichAsset]) -> None:
        managed_pairs: list[tuple[ImmichTag, str]] = [
            (tag, asset.id)
            for asset in assets
            for tag in asset.tags
            if is_managed_tag(tag.value)
        ]
        if not managed_pairs:
            return

        missing_values = [tag.value for tag, _ in managed_pairs if not tag.id]
        resolved_ids = self.ensure_tags(missing_values) if missing_values else {}

        assets_by_tag_id: dict[str, list[str]] = {}
        for tag, asset_id in managed_pairs:
            tag_id = tag.id or resolved_ids[tag.value]
            _append_unique(assets_by_tag_id, tag_id, asset_id)

        for tag_id, asset_ids in assets_by_tag_id.items():
            self.untag_assets(tag_id, asset_ids)

    def _list_albums(self) -> list[_AlbumSummaryPayload]:
        payload = self._request_json("GET", "/api/albums")
        if not isinstance(payload, list):
            raise ImmichApiError("Expected JSON list from GET /api/albums")
        return [_AlbumSummaryPayload.model_validate(item) for item in payload]

    def _get_album_detail(self, album_id: str) -> _AlbumDetailPayload:
        try:
            payload = self._request_json("GET", f"/api/albums/{album_id}")
        except ImmichApiError as exc:
            if exc.status_code == 404:
                raise AlbumNotFoundError(f"Album not found by id: {album_id!r}") from exc
            raise
        return _AlbumDetailPayload.model_validate(payload)

    @staticmethod
    def _match_album_by_name(
        albums: Sequence[_AlbumSummaryPayload], name: str
    ) -> str:
        matches = [album for album in albums if album.album_name == name]
        if not matches:
            raise AlbumNotFoundError(f"Album not found by name: {name!r}")
        if len(matches) > 1:
            candidate_ids = ", ".join(album.id for album in matches)
            raise AmbiguousAlbumError(
                f"Multiple albums match name {name!r}: {candidate_ids}"
            )
        return matches[0].id

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json, params=params)
        except httpx.HTTPError as exc:
            raise ImmichApiError(f"Immich request {method} {path} failed: {exc}") from exc

        if response.is_error:
            raise ImmichApiError(
                f"Immich API error {response.status_code} on {method} {path}: {response.text}",
                status_code=response.status_code,
            )

        return response

    def _request_json(self, method: str, path: str, *, json: Any = None) -> Any:
        response = self._send_request(method, path, json=json)

        if response.status_code == 204 or not response.content:
            return None

        return response.json()
