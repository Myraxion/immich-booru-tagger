"""Batch orchestration runner for album resolution, filtering, and tagging."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from immich_booru_tagger.config import AppConfig, RoutingMode
from immich_booru_tagger.immich import ImmichAsset, ImmichClient, TargetAlbum
from immich_booru_tagger.rules import (
    PROCESSED_MARKER,
    compute_dimension_meta_tags,
    normalize_booru_tag,
)

logger = logging.getLogger(__name__)


class VisionPipeline(Protocol):
    """Port reserved for vision-based Booru tag prediction."""

    def predict(
        self, asset: ImmichAsset, mode: RoutingMode
    ) -> Sequence[tuple[str, str]]:
        """Predict raw (category, tag_name) pairs for a single asset."""
        ...


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
    ) -> RunSummary:
        """Execute one batch tagging pass across configured Target Albums."""
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")

        if not dry_run:
            raise NotImplementedError(
                "Immich write-back is not implemented yet; run with --dry-run."
            )

        resolved_albums = self._client.resolve_target_albums(self._config.albums)
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
        for batch_index, batch in enumerate(batches, start=1):
            logger.info(
                "Processing micro-batch %d/%d (%d assets, dry_run=%s)",
                batch_index,
                len(batches),
                len(batch),
                dry_run,
            )
            for album, asset in batch:
                tags = self._compute_asset_tags(asset, album.mode)
                logger.info(
                    "[dry-run] asset=%s (%s) album=%s mode=%s tags=%s",
                    asset.id,
                    asset.original_file_name,
                    album.name,
                    album.mode,
                    tags,
                )
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

        return RunSummary(
            albums_resolved=len(resolved_albums),
            total_assets_scanned=total_scanned,
            total_eligible=total_eligible,
            processed_count=len(results),
            batches_count=len(batches),
            dry_run=dry_run,
            results=results,
        )

    def _compute_asset_tags(self, asset: ImmichAsset, mode: RoutingMode) -> list[str]:
        raw_tags: list[str] = []
        if self._vision_pipeline is not None:
            for category, tag_name in self._vision_pipeline.predict(asset, mode):
                raw_tags.append(normalize_booru_tag(category, tag_name))

        raw_tags.extend(compute_dimension_meta_tags(asset.width, asset.height))
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
