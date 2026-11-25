"""Batch orchestration runner for album resolution, filtering, and tagging."""

from __future__ import annotations

import io
import logging
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
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
            max_eligible=limit,
            force=force,
        )
        total_scanned, eligible_pairs = self._collect_eligible_assets(
            resolved_albums, force=force
        )
        total_eligible = len(eligible_pairs)

        selected = eligible_pairs[:limit] if limit is not None else eligible_pairs
        batch_size = self._config.batch.size
        total_batches = (len(selected) + batch_size - 1) // batch_size if selected else 0

        results: list[AssetTagResult] = []
        skipped_count = 0
        with (
            self._prefetch_thumbnails(selected, batch_size=batch_size) as prefetched,
            self._batch_writer(dry_run=dry_run, force=force) as submit_batch,
        ):
            batch_index = 0
            batch_updates: list[tuple[ImmichAsset, list[str]]] = []
            log_prefix = "[dry-run] " if dry_run else ""

            for idx, (album, asset, image) in enumerate(prefetched):
                if idx % batch_size == 0:
                    batch_index += 1
                    remaining = min(batch_size, len(selected) - idx)
                    logger.info(
                        "Processing micro-batch %d/%d (%d assets, dry_run=%s)",
                        batch_index,
                        total_batches,
                        remaining,
                        dry_run,
                    )

                if image is not None:
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
                else:
                    skipped_count += 1

                is_batch_end = (idx + 1) % batch_size == 0 or idx + 1 == len(selected)
                if is_batch_end and batch_updates:
                    submit_batch(batch_updates)
                    batch_updates = []

        return RunSummary(
            albums_resolved=len(resolved_albums),
            total_assets_scanned=total_scanned,
            total_eligible=total_eligible,
            processed_count=len(results),
            batches_count=total_batches,
            dry_run=dry_run,
            results=results,
            skipped_count=skipped_count,
        )

    @contextmanager
    def _batch_writer(
        self,
        *,
        dry_run: bool,
        force: bool,
        max_in_flight: int = 2,
    ) -> Iterator[Callable[[list[tuple[ImmichAsset, list[str]]]], None]]:
        if dry_run:
            yield lambda _batch: None
            return

        in_flight: deque[Future[None]] = deque()
        with ThreadPoolExecutor(max_workers=1) as executor:
            def _submit(batch_updates: list[tuple[ImmichAsset, list[str]]]) -> None:
                while in_flight and in_flight[0].done():
                    in_flight.popleft().result()
                while len(in_flight) >= max_in_flight:
                    in_flight.popleft().result()
                in_flight.append(
                    executor.submit(
                        self._client.sync_batch_tags,
                        batch_updates,
                        force=force,
                    )
                )

            try:
                yield _submit
            except BaseException:
                for future in in_flight:
                    future.cancel()
                raise

            while in_flight:
                in_flight.popleft().result()

    @contextmanager
    def _prefetch_thumbnails(
        self,
        selected: Sequence[tuple[TargetAlbum, ImmichAsset]],
        *,
        batch_size: int,
    ) -> Iterator[Iterator[tuple[TargetAlbum, ImmichAsset, Image.Image | None]]]:
        if not selected:
            yield iter(())
            return

        if len(selected) == 1:
            album, asset = selected[0]
            yield iter(((album, asset, self._load_thumbnail_image(asset)),))
            return

        window_size = min(max(batch_size, 4), 16, len(selected))
        max_workers = min(4, len(selected))
        pending: deque[tuple[TargetAlbum, ImmichAsset, Future[Image.Image | None]]] = deque()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            next_idx = 0
            while next_idx < window_size:
                album, asset = selected[next_idx]
                pending.append(
                    (album, asset, executor.submit(self._load_thumbnail_image, asset))
                )
                next_idx += 1

            def _iter_items() -> Iterator[tuple[TargetAlbum, ImmichAsset, Image.Image | None]]:
                nonlocal next_idx
                while pending:
                    album, asset, future = pending.popleft()
                    if next_idx < len(selected):
                        next_album, next_asset = selected[next_idx]
                        pending.append(
                            (
                                next_album,
                                next_asset,
                                executor.submit(self._load_thumbnail_image, next_asset),
                            )
                        )
                        next_idx += 1
                    yield album, asset, future.result()

            try:
                yield _iter_items()
            finally:
                for _, _, future in pending:
                    if not future.cancel():
                        try:
                            leftover = future.result()
                        except Exception:
                            leftover = None
                        if leftover is not None:
                            leftover.close()

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
