"""Batch orchestration runner for album resolution, filtering, and tagging."""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, UnidentifiedImageError

from immich_booru_tagger.config import AppConfig, RoutingMode
from immich_booru_tagger.immich import (
    ImmichApiError,
    ImmichAsset,
    ImmichClient,
    TargetAlbum,
)
from immich_booru_tagger.pipeline import VisionPipeline
from immich_booru_tagger.rules import (
    PROCESSED_MARKER,
    compute_dimension_meta_tags,
    extract_artist_tag,
    normalize_booru_tag,
)
from immich_booru_tagger.timefilter import validate_time_range

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssetTagResult:
    """Computed tags for an Eligible Asset."""

    asset_id: str
    file_name: str
    album_id: str
    album_name: str
    mode: RoutingMode
    tags: list[str]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregated outcome of a BatchRunner execution."""

    albums_resolved: int
    total_assets_scanned: int
    total_eligible: int
    processed_count: int
    batches_count: int
    dry_run: bool
    results: list[AssetTagResult]
    skipped_count: int = 0


class BatchRunner:
    """Coordinates Target Album resolution, Eligible Asset filtering, and Micro-batches."""

    def __init__(
        self,
        config: AppConfig,
        client: ImmichClient,
        *,
        vision_pipeline: VisionPipeline | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._vision_pipeline = vision_pipeline

    def run(
        self,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        force: bool = False,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> RunSummary:
        """Execute one batch tagging pass across configured Target Albums."""
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        validate_time_range(created_after, created_before)

        resolved_albums = self._client.resolve_target_albums(
            self._config.albums,
            created_after=created_after,
            created_before=created_before,
        )
        total_scanned, eligible_pairs = self._collect_eligible_assets(
            resolved_albums, force=force
        )
        total_eligible = len(eligible_pairs)

        selected = eligible_pairs[:limit] if limit is not None else eligible_pairs
        batch_size = self._config.batch.size
        batches = [
            selected[i : i + batch_size] for i in range(0, len(selected), batch_size)
        ]

        results: list[AssetTagResult] = []
        skipped_count = 0
        for batch_index, batch in enumerate(batches, start=1):
            logger.info(
                "Processing micro-batch %d/%d (%d assets, dry_run=%s)",
                batch_index,
                len(batches),
                len(batch),
                dry_run,
            )
            batch_updates: list[tuple[ImmichAsset, list[str]]] = []
            log_prefix = "[dry-run] " if dry_run else ""
            for album, asset in batch:
                image = self._load_thumbnail_image(asset)
                if image is None:
                    skipped_count += 1
                    continue
                try:
                    tags = self._compute_asset_tags(asset, image, album)
                finally:
                    image.close()

                logger.info(
                    "%sasset=%s (%s) album=%s mode=%s tags=%s",
                    log_prefix,
                    asset.id,
                    asset.original_file_name,
                    album.name,
                    album.mode,
                    tags,
                )
                batch_updates.append((asset, tags))
                results.append(
                    AssetTagResult(
                        asset_id=asset.id,
                        file_name=asset.original_file_name,
                        album_id=album.id,
                        album_name=album.name,
                        mode=album.mode,
                        tags=tags,
                    )
                )

            if not dry_run and batch_updates:
                self._client.sync_batch_tags(batch_updates, force=force)

        return RunSummary(
            albums_resolved=len(resolved_albums),
            total_assets_scanned=total_scanned,
            total_eligible=total_eligible,
            processed_count=len(results),
            batches_count=len(batches),
            dry_run=dry_run,
            results=results,
            skipped_count=skipped_count,
        )

    def _load_thumbnail_image(self, asset: ImmichAsset) -> Image.Image | None:
        try:
            raw_bytes = self._client.fetch_asset_thumbnail(
                asset.id, size=self._config.immich.thumbnail_size
            )
        except ImmichApiError as exc:
            if exc.status_code in (401, 403):
                raise
            logger.warning(
                "Skipping asset %s (%s): failed to fetch thumbnail: %s",
                asset.id,
                asset.original_file_name,
                exc,
            )
            return None

        image: Image.Image | None = None
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
            return image
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            EOFError,
            SyntaxError,
        ) as exc:
            if image is not None:
                image.close()
            logger.warning(
                "Skipping asset %s (%s): failed to decode thumbnail image: %s",
                asset.id,
                asset.original_file_name,
                exc,
            )
            return None

    def _compute_asset_tags(
        self, asset: ImmichAsset, image: Image.Image, album: TargetAlbum
    ) -> list[str]:
        raw_tags: list[str] = []
        if self._vision_pipeline is not None:
            for category, tag_name in self._vision_pipeline.predict(image, album.mode):
                raw_tags.append(normalize_booru_tag(category, tag_name))

        if self._config.pipeline.enable_dimension_tags:
            raw_tags.extend(compute_dimension_meta_tags(asset.width, asset.height))
        if artist_tag := extract_artist_tag(
            asset.original_file_name, album.artist_patterns
        ):
            raw_tags.append(artist_tag)
        raw_tags.extend(album.extra_tags)
        return list(dict.fromkeys(raw_tags))

    @staticmethod
    def _collect_eligible_assets(
        albums: Sequence[TargetAlbum], *, force: bool
    ) -> tuple[int, list[tuple[TargetAlbum, ImmichAsset]]]:
        seen_asset_ids: set[str] = set()
        eligible: list[tuple[TargetAlbum, ImmichAsset]] = []
        total_scanned = 0

        for album in albums:
            for asset in album.assets:
                if asset.id in seen_asset_ids:
                    continue
                seen_asset_ids.add(asset.id)
                total_scanned += 1

                if asset.type != "IMAGE":
                    continue
                if not force and PROCESSED_MARKER in asset.tag_values:
                    continue

                eligible.append((album, asset))

        return total_scanned, eligible
