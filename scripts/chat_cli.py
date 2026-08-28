#!/usr/bin/env python3
"""실행 중인 Chat API를 터미널에서 직접 확인하는 대화형 CLI."""

import argparse
from collections.abc import Callable, Sequence
from typing import Any, Optional

import httpx


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 180.0

Write = Callable[[str], None]
Read = Callable[[str], str]


def positive_float(value: str) -> float:
    """argparse에서 양수 timeout만 허용한다."""

    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 숫자를 입력해 주세요.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실행 중인 Riido Chat API의 멀티턴 흐름을 테스트합니다."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"FastAPI 서버 주소 (기본값: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "요청 제한 시간(초) "
            f"(기본값: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    return parser


class ChatCli:
    """conversationId를 기억하며 Chat API 요청과 출력을 담당한다."""

    def __init__(
        self,
        endpoint: str,
        client: httpx.Client,
        write: Write = print,
    ) -> None:
        self.endpoint = endpoint
        self.client = client
        self.write = write
        self.conversation_id: Optional[str] = None

    def reset(self) -> None:
        self.conversation_id = None
        self.write("새 대화를 시작할게.")

    def show_status(self) -> None:
        if self.conversation_id is None:
            self.write("현재 대화: 새 대화 (첫 질문 대기 중)")
            return
        self.write(f"현재 conversationId: {self.conversation_id}")

    def ask(self, question: str) -> None:
        payload = {"question": question}
        if self.conversation_id is not None:
            payload["conversationId"] = self.conversation_id

        try:
            response = self.client.post(self.endpoint, json=payload)
        except httpx.TimeoutException:
            self.write(
                "요청 시간이 초과됐어. 서버 로그를 확인하거나 "
                "--timeout 값을 늘려서 다시 실행해 줘."
            )
            return
        except httpx.ConnectError:
            self.write(
                "서버에 연결할 수 없어. FastAPI가 실행 중인지와 "
                f"주소({self.endpoint})를 확인해 줘."
            )
            return
        except httpx.HTTPError as exc:
            self.write(f"HTTP 요청에 실패했어: {exc}")
            return

        body = self._read_json(response)
        if body is None:
            return

        returned_conversation_id = body.get("conversationId")
        if isinstance(returned_conversation_id, str) and returned_conversation_id:
            self.conversation_id = returned_conversation_id

        self._render(response.status_code, body)

        error = body.get("error")
        error_code = error.get("code") if isinstance(error, dict) else None
        if response.status_code == 404 or error_code == "NOT_FOUND":
            self.conversation_id = None
            self.write("이어갈 수 없는 대화라서 다음 질문부터 새 대화로 시작해.")

    def _read_json(self, response: httpx.Response) -> Optional[dict[str, Any]]:
        try:
            body = response.json()
        except ValueError:
            preview = response.text.strip()[:500] or "(응답 본문 없음)"
            self.write(
                f"서버가 JSON이 아닌 응답을 보냈어 "
                f"(HTTP {response.status_code}): {preview}"
            )
            return None

        if not isinstance(body, dict):
            self.write(
                f"예상하지 못한 응답 형식이야 "
                f"(HTTP {response.status_code}): {body!r}"
            )
            return None
        return body

    def _render(self, status_code: int, body: dict[str, Any]) -> None:
        status = body.get("status")
        self.write("")

        if status == "COMPLETED":
            self.write("[COMPLETED]")
            self._write_identifiers(body)
            answer = body.get("answer")
            markdown = (
                answer.get("answerMarkdown")
                if isinstance(answer, dict)
                else None
            )
            self.write(str(markdown or "(답변 본문 없음)"))
            self._write_citations(body.get("citations"))
            return

        if status == "WITHHELD":
            self.write("[WITHHELD]")
            self._write_identifiers(body)
            withheld = body.get("withheld")
            if isinstance(withheld, dict):
                reason_code = withheld.get("reasonCode", "UNKNOWN")
                message = withheld.get("message", "답변이 보류됐어.")
                self.write(f"{reason_code}: {message}")
            else:
                self.write("답변이 보류됐어.")
            return

        if status == "ERROR":
            self.write(f"[ERROR / HTTP {status_code}]")
            self._write_identifiers(body)
            error = body.get("error")
            if isinstance(error, dict):
                code = error.get("code", "UNKNOWN")
                message = error.get("message", "알 수 없는 오류가 발생했어.")
                self.write(f"{code}: {message}")
            else:
                self.write("알 수 없는 오류가 발생했어.")
            return

        if status_code == 422:
            self.write("[VALIDATION ERROR / HTTP 422]")
            self._write_validation_errors(body.get("detail"))
            return

        self.write(f"[UNEXPECTED RESPONSE / HTTP {status_code}]")
        self.write(str(body))

    def _write_identifiers(self, body: dict[str, Any]) -> None:
        conversation_id = body.get("conversationId")
        rag_run_id = body.get("ragRunId")
        if conversation_id:
            self.write(f"conversationId: {conversation_id}")
        if rag_run_id:
            self.write(f"ragRunId: {rag_run_id}")
        self.write("")

    def _write_citations(self, citations: Any) -> None:
        if not isinstance(citations, list) or not citations:
            return

        self.write("")
        self.write("출처")
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            number = citation.get("citationNumber", "?")
            document_title = citation.get("documentTitle", "제목 없음")
            section_path = citation.get("sectionPath")
            path = [document_title]
            if isinstance(section_path, list):
                path.extend(str(section) for section in section_path)
            self.write(f"[{number}] {' > '.join(path)}")
            source_url = citation.get("sourceUrl")
            if source_url:
                self.write(f"    {source_url}")

    def _write_validation_errors(self, detail: Any) -> None:
        if not isinstance(detail, list):
            self.write(str(detail or "요청 값이 올바르지 않아."))
            return

        for item in detail:
            if not isinstance(item, dict):
                self.write(str(item))
                continue
            location = item.get("loc", [])
            field = ".".join(str(part) for part in location)
            message = item.get("msg", "올바르지 않은 값")
            self.write(f"- {field}: {message}")


def run_interactive(
    chat: ChatCli,
    read: Read = input,
    write: Write = print,
) -> None:
    write("Riido Multi-turn CLI")
    write(f"API: {chat.endpoint}")
    write("명령어: /new 새 대화 | /status 현재 ID | /quit 종료")

    while True:
        try:
            raw = read("\nyou> ")
        except EOFError:
            write("\nCLI를 종료할게.")
            return
        except KeyboardInterrupt:
            write("\nCLI를 종료할게.")
            return

        question = raw.strip()
        if not question:
            continue

        command = question.lower()
        if command in {"/quit", "/exit"}:
            write("CLI를 종료할게.")
            return
        if command == "/new":
            chat.reset()
            continue
        if command == "/status":
            chat.show_status()
            continue
        if command == "/help":
            write("명령어: /new 새 대화 | /status 현재 ID | /quit 종료")
            continue

        chat.ask(question)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    endpoint = f"{args.base_url.rstrip('/')}/api/chat"
    headers = {"Accept": "application/json"}
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        chat = ChatCli(endpoint=endpoint, client=client)
        run_interactive(chat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
