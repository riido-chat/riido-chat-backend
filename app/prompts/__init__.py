"""모델에 보내는 프롬프트 본문의 버전 파일과 지문 표.

본문은 `<component>/<version>.md` 파일에 두고 추가만 한다. 호출부는 `get(component, version)`
으로 현재 버전의 본문을 받는다. 어떤 버전을 쓸지는 아직 호출부의 버전 상수가 정한다.
"""

from app.prompts.registry import PromptNotFoundError, components, get, versions

__all__ = ["PromptNotFoundError", "components", "get", "versions"]
