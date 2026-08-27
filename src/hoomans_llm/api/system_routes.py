"""Health and provider model discovery routes."""

from fastapi import APIRouter, Request

from hoomans_llm.api.models import HealthResponse, ModelInfo, ModelListResponse

from .route_support import _bridge_status, _registry

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    registry = _registry(request)
    return HealthResponse(
        service=request.app.title,
        providers=list(registry.provider_names),
        bridge=_bridge_status(request),
    )


@router.get("/v1/models", response_model=ModelListResponse, tags=["models"])
async def list_models(request: Request) -> ModelListResponse:
    registry = _registry(request)
    return ModelListResponse(
        data=[
            ModelInfo(id=model_id, owned_by=provider_name)
            for provider_name, model_id in registry.model_ids()
        ]
    )
