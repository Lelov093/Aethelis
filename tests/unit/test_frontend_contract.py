from __future__ import annotations

import json
from pathlib import Path

from aethelis.api.app import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_api_manifest_matches_fastapi_routes() -> None:
    manifest = json.loads((REPOSITORY_ROOT / "frontend" / "api-contract.json").read_text())
    app = create_app(
        command_service=object(),  # type: ignore[arg-type]
        product_service=object(),  # type: ignore[arg-type]
        principal_resolver=object(),  # type: ignore[arg-type]
        projection_service=object(),  # type: ignore[arg-type]
    )
    openapi = app.openapi()

    for operation in manifest["operations"]:
        assert operation["path"] in openapi["paths"]
        assert operation["method"] in openapi["paths"][operation["path"]]

    schemas = openapi["components"]["schemas"]
    assert {"id", "display_name", "locale"}.issubset(schemas["PlayerProfile"]["properties"])
    assert {"name", "status", "latest_save"}.issubset(schemas["WorldTimelineView"]["properties"])
    assert {"name", "world_version", "reason"}.issubset(schemas["SavePointView"]["properties"])
    assert {"resources", "opportunities", "knowledge", "relationships", "commitments"}.issubset(
        schemas["JournalView"]["properties"]
    )


def test_player_client_uses_connected_projection_and_command_routes() -> None:
    client = (REPOSITORY_ROOT / "frontend" / "src" / "api" / "client.ts").read_text(
        encoding="utf-8"
    )
    play_view = (
        REPOSITORY_ROOT / "frontend" / "src" / "features" / "play" / "PlayView.tsx"
    ).read_text(encoding="utf-8")

    for route in ("/scene", "/map", "/journal", "/resume-summary", "/commands"):
        assert route in client
    assert "Idempotency-Key" in client
    assert "pollCommand" in play_view
    assert "cancelCommand" in play_view
    assert "expected_world_version" in play_view


def test_mistgate_work_block_two_assets_are_manifested_and_referenced() -> None:
    asset_root = REPOSITORY_ROOT / "frontend" / "public" / "assets" / "mistgate"
    manifest = json.loads((asset_root / "asset-manifest.json").read_text(encoding="utf-8"))
    play_view = (
        REPOSITORY_ROOT / "frontend" / "src" / "features" / "play" / "PlayView.tsx"
    ).read_text(encoding="utf-8")

    assert manifest["status"] == "approved_for_product_use"
    assert len(manifest["assets"]) == 7
    assert {asset["subject_id"] for asset in manifest["assets"]} == {
        "council_square",
        "central_archive",
        "market_row",
        "workshop_lane",
        "old_aqueduct",
        "mira",
        "rowan",
    }
    for asset in manifest["assets"]:
        relative_uri = asset["uri"].removeprefix("/assets/mistgate/")
        assert (asset_root / relative_uri).is_file()
        assert asset["uri"] in play_view
