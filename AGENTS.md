# Repository Development Guide

## 프로젝트 목표와 기술 방향

- 기업 공식 이용가이드를 근거로 답하는 독립형 RAG 챗봇 백엔드를 만든다.
- 현재 입력은 GitBook Markdown을 우선 지원하되, 특정 문서 포맷에 종속되지 않도록 처리 단계별 책임을 분리한다.
- 목표 기술 구성은 FastAPI, PostgreSQL + pgvector, SSE, Kiwi BM25 + Vector Hybrid Search다.
- Reranker는 MVP 이후에 검토하며 현재 구현하지 않는다.

## 문서 처리 모델

- 아래 구조는 현재 구현 완료 상태가 아니라 프로젝트가 지향하는 책임 경계이며, 각 단계는 MVP 진행에 따라 순차적으로 구현한다.
- 파이프라인의 책임 경계는 `Loader → Normalizer → Sectioner → Chunker → Embedder`로 유지한다.
- 데이터 계층은 `Document → Section → Chunk`로 구분하고 각 계층의 식별자와 책임을 보존한다.
- Markdown에서는 H2를 기본 Section 경계로 사용하지만, Section 개념 자체를 H2에 종속시키지 않는다.
- Section의 추가 분할이 필요한 경우 H3, `details`, step, list 등 실제 문서 구조와 의미 경계를 활용한다.
- 짧더라도 의미적으로 독립된 Section은 길이만을 이유로 다른 Section과 병합하지 않는다.

## 설계와 확장성

- 교체 가능성이 높은 영역은 인터페이스와 책임 경계를 명확히 하되, 미래 기능의 실제 구현을 미리 만들지 않는다.
- 향후 교체나 확장을 어렵게 만드는 결합을 피한다.
- 새 추상화나 인터페이스를 추가하기 전에 실제로 교체 가능성이 있는 책임인지 확인한다.
- 필요가 입증되지 않은 Factory, Base, Manager 계층 등 과도한 추상화를 만들지 않는다.

## 검색과 답변 정책

- 검색된 공식 이용가이드 문서만 답변 근거로 사용한다.
- 근거가 없으면 내용을 임의로 생성하지 않는다.
- 답변에는 실제로 사용한 출처 1~3개를 제공한다.
- 출처는 최소 `Document > Section` 수준까지 식별하고 원본 페이지 URL을 사용한다.

## 멀티턴 정책

- 사용자 입력을 `FOLLOW_UP`과 `NEW_TOPIC`으로 구분한다.
- `FOLLOW_UP`일 때만 필요한 이전 문맥을 선별해 검색용 Query Rewrite에 사용한다.
- 전체 대화를 무조건 검색 질의에 포함하지 않는다.

## 색인 정책

- MVP에서는 문서 변경 시 전체 수집, 정제, 청킹, 임베딩을 다시 수행한다.
- `Document / Section / Chunk`의 식별자와 책임 경계는 향후 증분 재색인이 가능하도록 유지한다.
- 증분 재색인 자체는 현재 구현하지 않는다.

## 현재 MVP 범위

포함:

- GitBook Markdown 기반 문서 파이프라인
- 구조 기반 Section / Chunk 생성
- Kiwi BM25 + Vector Hybrid Search
- FastAPI + SSE
- 기본 Multi-turn
- Retrieval / Generation / Feedback 로그
- 전체 재색인

제외:

- PDF / HTML 완전 지원
- 증분 재색인
- Reranker
- Semantic Cache
- 사용자 개인화
- 문서 버전 이력 관리

## 개발 원칙

- MVP 완성을 우선한다.
- 요청받지 않은 대규모 리팩터링이나 관련 없는 코드 수정을 하지 않는다.
- 수정 전에 기존 코드 구조와 관례를 먼저 확인한다.
- 구현 후 변경 범위와 관련된 테스트를 실행한다.
- 코드와 문서가 충돌하면 실제 코드 상태를 먼저 확인하며 추측으로 맞추지 않는다.
