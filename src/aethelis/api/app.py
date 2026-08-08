from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from aethelis.api.auth import PrincipalContextResolver
from aethelis.product.command_contracts import (
    CommandInputMode,
    CommandReceipt,
    SubmitPlayerCommand,
)
from aethelis.product.command_service import CommandApplicationService
from aethelis.product.content_contracts import AvailableWorldContent
from aethelis.product.contracts import (
    PlayerProfile,
    PlaySession,
    PrincipalContext,
    ResumeState,
    SavePoint,
    WorldInstance,
)
from aethelis.product.errors import (
    ProductAccessDeniedError,
    ProductApplicationError,
    ProductConflictError,
    ProductNotFoundError,
)
from aethelis.product.projection_contracts import (
    JournalView,
    MapView,
    ResumeSummaryView,
    SavePointView,
    SceneView,
    WorldTimelineView,
)
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService


class SubmitCommandBody(BaseModel):
    player_profile_id: str = Field(min_length=1)
    play_session_id: str = Field(min_length=1)
    input_mode: CommandInputMode
    action_id: str | None = None
    text: str | None = Field(default=None, max_length=2000)
    actor_id: str = Field(min_length=1)
    target_ids: tuple[str, ...] = ()
    target_hints: dict[str, str] = Field(default_factory=dict)
    dialogue_interaction_id: str | None = Field(default=None, min_length=1, max_length=160)
    location_id: str | None = None
    client_scene_id: str | None = None
    expected_world_version: int = Field(ge=0)
    locale: str = Field(default="en", min_length=2, max_length=35)

    @model_validator(mode="after")
    def validate_input(self) -> "SubmitCommandBody":
        if self.input_mode == CommandInputMode.CONTEXTUAL_ACTION and self.action_id is None:
            raise ValueError("contextual action requires action_id")
        if self.input_mode == CommandInputMode.NATURAL_LANGUAGE_INTENT and not self.text:
            raise ValueError("natural-language intent requires text")
        return self


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    code: str
    detail: str


class CreatePackagedWorldBody(BaseModel):
    content_version_id: str = Field(min_length=1)
    player_profile_id: str = Field(min_length=1)
    name: str = Field(default="新的雾门时间线", min_length=1, max_length=120)


class StartSessionBody(BaseModel):
    player_profile_id: str = Field(min_length=1)


class CreateSaveBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    play_session_id: str | None = None


class ForkSaveBody(BaseModel):
    player_profile_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)


def create_app(
    *,
    command_service: CommandApplicationService,
    product_service: ProductApplicationService,
    principal_resolver: PrincipalContextResolver,
    projection_service: ProjectionService,
    allowed_origins: tuple[str, ...] = (),
) -> FastAPI:
    app = FastAPI(title="Aethelis Product API", version="1.0.0")
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Idempotency-Key"],
        )
    bearer = HTTPBearer(auto_error=False)

    def current_principal(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> PrincipalContext:
        token = (
            credentials.credentials
            if credentials is not None and credentials.scheme.lower() == "bearer"
            else None
        )
        if token is None and getattr(principal_resolver, "requires_bearer", True):
            raise ProductAccessDeniedError(
                "missing_access_token", "Bearer access token is required."
            )
        return principal_resolver.resolve(token)

    @app.exception_handler(ProductApplicationError)
    async def product_error_handler(
        _request: Request, exc: ProductApplicationError
    ) -> JSONResponse:
        http_status = _status_for_error(exc)
        problem = ProblemDetail(
            type=f"urn:aethelis:problem:{exc.code}",
            title="Aethelis request failed",
            status=http_status,
            code=exc.code,
            detail=str(exc),
        )
        return JSONResponse(status_code=http_status, content=problem.model_dump())

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/me", response_model=PlayerProfile)
    def get_player_profile(
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> PlayerProfile:
        return product_service.get_player_profile(principal=principal)

    @app.get(
        "/api/v1/world-definitions",
        response_model=tuple[AvailableWorldContent, ...],
    )
    def list_world_definitions(
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> tuple[AvailableWorldContent, ...]:
        return product_service.list_available_world_content(principal=principal)

    @app.post(
        "/api/v1/world-instances",
        response_model=ResumeState,
        status_code=status.HTTP_201_CREATED,
    )
    def create_world_instance(
        body: CreatePackagedWorldBody,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> ResumeState:
        return product_service.create_world_instance_from_content(
            principal=principal,
            content_version_id=body.content_version_id,
            player_profile_id=body.player_profile_id,
            name=body.name,
        )

    @app.get("/api/v1/world-instances", response_model=tuple[WorldTimelineView, ...])
    def list_world_instances(
        principal: Annotated[PrincipalContext, Depends(current_principal)],
        include_archived: bool = False,
    ) -> tuple[WorldTimelineView, ...]:
        return product_service.list_world_timelines(
            principal=principal,
            include_archived=include_archived,
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/saves",
        response_model=tuple[SavePointView, ...],
    )
    def list_saves(
        world_instance_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> tuple[SavePointView, ...]:
        return product_service.list_save_points(
            principal=principal,
            world_instance_id=world_instance_id,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/saves",
        response_model=SavePoint,
        status_code=status.HTTP_201_CREATED,
    )
    def create_save(
        world_instance_id: str,
        body: CreateSaveBody,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> SavePoint:
        return product_service.create_save_point(
            principal=principal,
            world_instance_id=world_instance_id,
            play_session_id=body.play_session_id,
            name=body.name,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/saves/{save_point_id}/fork",
        response_model=ResumeState,
        status_code=status.HTTP_201_CREATED,
    )
    def fork_save(
        world_instance_id: str,
        save_point_id: str,
        body: ForkSaveBody,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> ResumeState:
        return product_service.fork_world_from_save(
            principal=principal,
            player_profile_id=body.player_profile_id,
            source_world_instance_id=world_instance_id,
            save_point_id=save_point_id,
            name=body.name,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/archive",
        response_model=WorldInstance,
    )
    def archive_world(
        world_instance_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> WorldInstance:
        return product_service.archive_world_instance(
            principal=principal,
            world_instance_id=world_instance_id,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/restore",
        response_model=WorldInstance,
    )
    def restore_world(
        world_instance_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> WorldInstance:
        return product_service.restore_world_instance(
            principal=principal,
            world_instance_id=world_instance_id,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/sessions",
        response_model=PlaySession,
    )
    def start_or_resume_session(
        world_instance_id: str,
        body: StartSessionBody,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> PlaySession:
        return product_service.start_or_resume_session(
            principal=principal,
            world_instance_id=world_instance_id,
            player_profile_id=body.player_profile_id,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/commands",
        response_model=CommandReceipt,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit_command(
        world_instance_id: str,
        body: SubmitCommandBody,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
    ) -> CommandReceipt:
        return command_service.submit(
            principal=principal,
            world_instance_id=world_instance_id,
            request=SubmitPlayerCommand(idempotency_key=idempotency_key, **body.model_dump()),
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/commands/{command_id}",
        response_model=CommandReceipt,
    )
    def get_command(
        world_instance_id: str,
        command_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> CommandReceipt:
        return command_service.get(
            principal=principal,
            world_instance_id=world_instance_id,
            command_id=command_id,
        )

    @app.post(
        "/api/v1/world-instances/{world_instance_id}/commands/{command_id}/cancel",
        response_model=CommandReceipt,
    )
    def cancel_command(
        world_instance_id: str,
        command_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> CommandReceipt:
        return command_service.cancel(
            principal=principal,
            world_instance_id=world_instance_id,
            command_id=command_id,
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/scene",
        response_model=SceneView,
    )
    def get_scene(
        world_instance_id: str,
        player_profile_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> SceneView:
        return projection_service.scene(
            principal=principal,
            world_instance_id=world_instance_id,
            player_profile_id=player_profile_id,
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/map",
        response_model=MapView,
    )
    def get_map(
        world_instance_id: str,
        player_profile_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> MapView:
        return projection_service.map(
            principal=principal,
            world_instance_id=world_instance_id,
            player_profile_id=player_profile_id,
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/journal",
        response_model=JournalView,
    )
    def get_journal(
        world_instance_id: str,
        player_profile_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> JournalView:
        return projection_service.journal(
            principal=principal,
            world_instance_id=world_instance_id,
            player_profile_id=player_profile_id,
        )

    @app.get(
        "/api/v1/world-instances/{world_instance_id}/resume-summary",
        response_model=ResumeSummaryView,
    )
    def get_resume_summary(
        world_instance_id: str,
        player_profile_id: str,
        principal: Annotated[PrincipalContext, Depends(current_principal)],
    ) -> ResumeSummaryView:
        return projection_service.resume_summary(
            principal=principal,
            world_instance_id=world_instance_id,
            player_profile_id=player_profile_id,
        )

    return app


def _status_for_error(exc: ProductApplicationError) -> int:
    if exc.code == "command_rate_limited":
        return status.HTTP_429_TOO_MANY_REQUESTS
    if isinstance(exc, ProductAccessDeniedError):
        authentication_codes = {
            "missing_access_token",
            "invalid_access_token",
            "principal_not_active",
        }
        return (
            status.HTTP_401_UNAUTHORIZED
            if exc.code in authentication_codes
            else status.HTTP_403_FORBIDDEN
        )
    if isinstance(exc, ProductNotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, ProductConflictError):
        return status.HTTP_409_CONFLICT
    return status.HTTP_400_BAD_REQUEST
