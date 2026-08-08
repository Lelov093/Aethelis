from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from aethelis.api.app import create_app
from aethelis.product.command_contracts import (
    CommandExecution,
    CommandReceipt,
    PlayerCommand,
)
from aethelis.product.contracts import PlayerProfile, PrincipalContext, PrincipalRole

NOW = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)


class Resolver:
    def resolve(self, token: str) -> PrincipalContext:
        if token != "valid-token":
            raise ValueError("invalid")
        return PrincipalContext(principal_id="principal_1", roles=(PrincipalRole.PLAYER,))


class LocalResolver:
    requires_bearer = False

    def resolve(self, token: str | None) -> PrincipalContext:
        assert token is None
        return PrincipalContext(principal_id="principal_1", roles=(PrincipalRole.PLAYER,))


class Commands:
    def submit(self, *, principal, world_instance_id, request):
        command = PlayerCommand(
            id="command_1",
            principal_id=principal.principal_id,
            world_instance_id=world_instance_id,
            idempotency_key=request.idempotency_key,
            player_profile_id=request.player_profile_id,
            play_session_id=request.play_session_id,
            input_mode=request.input_mode,
            action_id=request.action_id,
            text=request.text,
            actor_id=request.actor_id,
            expected_world_version=request.expected_world_version,
            locale=request.locale,
            submitted_at=NOW,
            updated_at=NOW,
        )
        return CommandReceipt(
            command=command,
            execution=CommandExecution(
                command_id=command.id,
                attempt_count=0,
                created_at=NOW,
                updated_at=NOW,
            ),
            status_url=f"/api/v1/world-instances/{world_instance_id}/commands/{command.id}",
        )

    def get(self, **_kwargs):
        raise AssertionError("not used")

    def cancel(self, **_kwargs):
        raise AssertionError("not used")


class Projections:
    def scene(self, **_kwargs):
        raise AssertionError("not used")

    def resume_summary(self, **_kwargs):
        raise AssertionError("not used")

    def map(self, **_kwargs):
        raise AssertionError("not used")

    def journal(self, **_kwargs):
        raise AssertionError("not used")


class ProductService:
    def list_available_world_content(self, **_kwargs):
        return ()

    def create_world_instance_from_content(self, **_kwargs):
        raise AssertionError("not used")

    def start_or_resume_session(self, **_kwargs):
        raise AssertionError("not used")

    def get_player_profile(self, **_kwargs):
        return PlayerProfile(
            id="profile_1",
            principal_id="principal_1",
            display_name="雾门旅人",
            locale="zh-CN",
            created_at=NOW,
            updated_at=NOW,
        )

    def list_world_timelines(self, **_kwargs):
        return ()

    def list_save_points(self, **_kwargs):
        return ()

    def create_save_point(self, **_kwargs):
        raise AssertionError("not used")

    def fork_world_from_save(self, **_kwargs):
        raise AssertionError("not used")

    def archive_world_instance(self, **_kwargs):
        raise AssertionError("not used")

    def restore_world_instance(self, **_kwargs):
        raise AssertionError("not used")


def test_submit_command_requires_bearer_and_idempotency_key() -> None:
    client = TestClient(
        create_app(
            command_service=Commands(),
            product_service=ProductService(),
            principal_resolver=Resolver(),
            projection_service=Projections(),
        )
    )
    payload = {
        "player_profile_id": "profile_1",
        "play_session_id": "session_1",
        "input_mode": "contextual_action",
        "action_id": "inspect_archive",
        "actor_id": "player_1",
        "expected_world_version": 0,
    }

    unauthorized = client.post(
        "/api/v1/world-instances/world_1/commands",
        json=payload,
        headers={"Idempotency-Key": "request-0001"},
    )
    accepted = client.post(
        "/api/v1/world-instances/world_1/commands",
        json=payload,
        headers={"Authorization": "Bearer valid-token", "Idempotency-Key": "request-0001"},
    )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["code"] == "missing_access_token"
    assert accepted.status_code == 202
    assert accepted.json()["command"]["status"] == "submitted"
    assert accepted.json()["status_url"].endswith("/commands/command_1")


def test_openapi_exposes_versioned_command_polling_and_cancel_routes() -> None:
    client = TestClient(
        create_app(
            command_service=Commands(),
            product_service=ProductService(),
            principal_resolver=Resolver(),
            projection_service=Projections(),
        )
    )
    paths = client.get("/openapi.json").json()["paths"]
    base = "/api/v1/world-instances/{world_instance_id}/commands"
    assert "post" in paths[base]
    assert "get" in paths[f"{base}/{{command_id}}"]
    assert "post" in paths[f"{base}/{{command_id}}/cancel"]
    assert "get" in paths["/api/v1/world-instances/{world_instance_id}/scene"]
    assert "get" in paths["/api/v1/world-instances/{world_instance_id}/map"]
    assert "get" in paths["/api/v1/world-instances/{world_instance_id}/journal"]
    assert "get" in paths["/api/v1/world-instances/{world_instance_id}/resume-summary"]
    assert "post" in paths["/api/v1/world-instances"]
    assert "get" in paths["/api/v1/world-definitions"]
    assert "post" in paths["/api/v1/world-instances/{world_instance_id}/sessions"]
    assert "get" in paths["/api/v1/me"]
    assert "get" in paths["/api/v1/world-instances"]
    save_base = "/api/v1/world-instances/{world_instance_id}/saves"
    assert {"get", "post"}.issubset(paths[save_base])
    assert "post" in paths[f"{save_base}/{{save_point_id}}/fork"]
    assert "post" in paths["/api/v1/world-instances/{world_instance_id}/archive"]
    assert "post" in paths["/api/v1/world-instances/{world_instance_id}/restore"]


def test_local_single_user_api_requires_no_bearer_and_uses_exact_cors_origin() -> None:
    client = TestClient(
        create_app(
            command_service=Commands(),
            product_service=ProductService(),
            principal_resolver=LocalResolver(),
            projection_service=Projections(),
            allowed_origins=("http://localhost:5173",),
        )
    )

    response = client.get(
        "/api/v1/me",
        headers={"Origin": "http://localhost:5173"},
    )
    rejected_origin = client.options(
        "/api/v1/me",
        headers={
            "Origin": "http://192.168.1.8:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "雾门旅人"
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert rejected_origin.status_code == 400
    assert "access-control-allow-origin" not in rejected_origin.headers
