from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health", summary="애플리케이션 상태 확인")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
