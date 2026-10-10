"""모델에 보내는 프롬프트 본문의 버전 파일과 지문 표.

본문은 `<component>/<version>.md` 파일에 두고 추가만 한다. 채팅 턴에서 어떤 버전을 쓸지는 그 턴의
프로필 버전 구성 행이 정하고(app/profiles/stage_settings.py), 호출부의 버전 상수는 프로필 버전 없이
부르는 경로(평가 도구 등)의 코드 기본값이다.
"""

from app.prompts.registry import PromptNotFoundError, components, get, versions

__all__ = ["PromptNotFoundError", "components", "get", "versions"]
