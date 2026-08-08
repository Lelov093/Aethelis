from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aethelis.schemas.metadata import MetadataSeed
from aethelis.schemas.seed import AgentsSeed, BeliefsSeed, MemoriesSeed, SeedBundle, SeedManifest
from aethelis.schemas.world import WorldState
from aethelis.seeds.validator import SeedValidationError

REQUIRED_SEED_FILES = {
    "manifest": "seed_manifest.yaml",
    "world": "world.yaml",
    "agents": "agents.yaml",
    "beliefs": "beliefs.yaml",
    "memories": "memories.yaml",
}
OPTIONAL_SEED_FILES = {
    "metadata": "metadata.yaml",
}


@dataclass(frozen=True)
class SeedLoadResult:
    seed_path: Path
    bundle: SeedBundle | None
    errors: tuple[SeedValidationError, ...]
    loaded_files: tuple[str, ...]


class SeedLoader:
    def load(self, seed_path: Path) -> SeedLoadResult:
        seed_path = seed_path.resolve()
        errors: list[SeedValidationError] = []
        loaded: dict[str, Any] = {}
        loaded_files: list[str] = []

        if not seed_path.exists() or not seed_path.is_dir():
            return SeedLoadResult(
                seed_path=seed_path,
                bundle=None,
                loaded_files=(),
                errors=(
                    SeedValidationError(
                        file=str(seed_path),
                        object_id=None,
                        field_path="seed_path",
                        error_type="missing_seed_directory",
                        message="Seed path does not exist or is not a directory.",
                    ),
                ),
            )

        for key, filename in REQUIRED_SEED_FILES.items():
            file_path = seed_path / filename
            if not file_path.exists():
                errors.append(
                    SeedValidationError(
                        file=filename,
                        object_id=None,
                        field_path=key,
                        error_type="missing_file",
                        message=f"Required seed file is missing: {filename}",
                    )
                )
                continue
            try:
                loaded[key] = _read_yaml(file_path)
                loaded_files.append(filename)
            except yaml.YAMLError as exc:
                errors.append(
                    SeedValidationError(
                        file=filename,
                        object_id=None,
                        field_path=key,
                        error_type="yaml_parse_error",
                        message=str(exc),
                    )
                )

        for key, filename in OPTIONAL_SEED_FILES.items():
            file_path = seed_path / filename
            if file_path.exists():
                try:
                    loaded[key] = _read_yaml(file_path)
                    loaded_files.append(filename)
                except yaml.YAMLError as exc:
                    errors.append(
                        SeedValidationError(
                            file=filename,
                            object_id=None,
                            field_path=key,
                            error_type="yaml_parse_error",
                            message=str(exc),
                        )
                    )

        if errors:
            return SeedLoadResult(seed_path, None, tuple(errors), tuple(loaded_files))

        try:
            bundle = SeedBundle(
                manifest=SeedManifest.model_validate(loaded["manifest"]),
                world=WorldState.model_validate(loaded["world"]),
                agents=AgentsSeed.model_validate(loaded["agents"]),
                beliefs=BeliefsSeed.model_validate(loaded["beliefs"]),
                memories=MemoriesSeed.model_validate(loaded["memories"]),
                metadata=(
                    MetadataSeed.model_validate(loaded["metadata"])
                    if "metadata" in loaded
                    else None
                ),
            )
        except ValidationError as exc:
            for error in exc.errors(include_url=False):
                errors.append(
                    SeedValidationError(
                        file="seed",
                        object_id=_object_id_from_error(error.get("input")),
                        field_path=".".join(str(part) for part in error["loc"]),
                        error_type=str(error["type"]),
                        message=str(error["msg"]),
                    )
                )
            return SeedLoadResult(seed_path, None, tuple(errors), tuple(loaded_files))

        return SeedLoadResult(seed_path, bundle, (), tuple(loaded_files))


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    return value


def _object_id_from_error(value: object) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return value["id"]
    return None
