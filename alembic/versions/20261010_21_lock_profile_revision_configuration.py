"""Lock chat profile revision configuration with triggers.

프로필 버전의 구성 칸은 생성 시점부터 불변이고 상태 칸만 바뀐다(#211 2단계, D12). 앱 코드가 아니라
DB 트리거로 강제해 손 수정, 운영 스크립트, 새 경로 모두 같은 규칙을 따른다.

chat_profile_revisions
- UPDATE: status 외 칸은 바꿀 수 없다. 칸을 이름으로 나열하지 않고 행 전체에서 status 만 빼고
  비교하므로 나중에 더하는 칸도 구성 칸으로 잠긴다.
- 상태 전이: DRAFT→TESTING, TESTING→PUBLISHED, PUBLISHED→RETIRED, TESTING→RETIRED,
  RETIRED→PUBLISHED 만 허용한다. 같은 상태로 쓰는 것은 변경이 아니라 통과한다.
  DRAFT→RETIRED 는 막는다. 서빙된 적 없는 DRAFT 는 지우면 되고, 열어 두면
  DRAFT→RETIRED→PUBLISHED 로 TESTING 을 건너뛰는 길이 생긴다.
- INSERT: 상태 제한 없음(운영 부트스트랩은 PUBLISHED 로 바로 만든다). 만든 id 를 트랜잭션 범위
  설정값(riido.created_profile_revision_ids)에 적어 같은 트랜잭션의 구성 행 INSERT 를 허용한다.
- DELETE: DRAFT 이고 conversations.chat_profile_revision_id, rag_runs.profile_revision_id 가
  가리키지 않을 때만. 외래키(RESTRICT)도 막지만 규칙 이름이 드러나게 먼저 거부한다.

profile_revision_components
- UPDATE: 전부 거부.
- INSERT: 부모 프로필 버전이 DRAFT 이거나 같은 트랜잭션에서 만든 프로필 버전일 때만.
- DELETE: 부모 프로필 버전이 DRAFT 일 때만.
- 부모 행은 FOR SHARE 로 읽어, 동시에 진행하는 상태 전이와 구성 행 변경이 엇갈리지 않게 한다.

거부는 SQLSTATE 23001(restrict_violation), 상태 전이만 23514(check_violation)이고, 메시지에
규칙과 칸 이름을, 오류 필드(TABLE, COLUMN, CONSTRAINT)에 대상을 담는다.

행 트리거라 TRUNCATE 와 트리거를 끈 세션(session_replication_role = replica)은 막지 않는다.
기존 행은 검사하지 않는다(규칙은 이후 변경에만 적용).

downgrade 는 트리거와 함수를 지운다.

Revision ID: 20261010_21
Revises: 20261010_20
Create Date: 2026-10-10
"""

from typing import Optional, Sequence, Union

from alembic import op


revision: str = "20261010_21"
down_revision: Optional[str] = "20261010_20"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


# 같은 트랜잭션에서 만든 프로필 버전 id 목록(쉼표 구분). set_config(..., true) 라 트랜잭션이 끝나면 사라진다.
CREATED_IDS_SETTING = "riido.created_profile_revision_ids"

# (트리거 이름, 테이블, 시점과 동작, 함수)
TRIGGERS = (
    (
        "trg_chat_profile_revisions_lock_update",
        "chat_profile_revisions",
        "BEFORE UPDATE",
        "chat_profile_revision_lock_update",
    ),
    (
        "trg_chat_profile_revisions_lock_delete",
        "chat_profile_revisions",
        "BEFORE DELETE",
        "chat_profile_revision_lock_delete",
    ),
    (
        "trg_chat_profile_revisions_note_insert",
        "chat_profile_revisions",
        "AFTER INSERT",
        "chat_profile_revision_note_insert",
    ),
    (
        "trg_profile_revision_components_lock",
        "profile_revision_components",
        "BEFORE INSERT OR UPDATE OR DELETE",
        "profile_revision_component_lock",
    ),
)

FUNCTIONS = {
    "chat_profile_revision_lock_update": r"""
DECLARE
    changed text[];
BEGIN
    SELECT array_agg(n.key ORDER BY n.key) INTO changed
    FROM jsonb_each(to_jsonb(NEW)) AS n
    JOIN jsonb_each(to_jsonb(OLD)) AS o USING (key)
    WHERE n.key <> 'status' AND n.value IS DISTINCT FROM o.value;
    IF changed IS NOT NULL THEN
        RAISE EXCEPTION
            '프로필 버전 구성 칸은 바꿀 수 없습니다(상태 칸만 변경): chat_profile_revisions.id=%, 칸=%',
            OLD.id, array_to_string(changed, ', ')
            USING ERRCODE = 'restrict_violation',
                  TABLE = TG_TABLE_NAME,
                  COLUMN = changed[1],
                  CONSTRAINT = 'profile_revision_configuration_immutable',
                  HINT = '고치려면 새 프로필 버전을 만듭니다.';
    END IF;
    IF NEW.status IS DISTINCT FROM OLD.status AND (OLD.status, NEW.status) NOT IN (
        ('DRAFT', 'TESTING'),
        ('TESTING', 'PUBLISHED'),
        ('PUBLISHED', 'RETIRED'),
        ('TESTING', 'RETIRED'),
        ('RETIRED', 'PUBLISHED')
    ) THEN
        RAISE EXCEPTION
            '허용하지 않는 프로필 버전 상태 전이입니다: chat_profile_revisions.id=%, % → %',
            OLD.id, OLD.status, NEW.status
            USING ERRCODE = 'check_violation',
                  TABLE = TG_TABLE_NAME,
                  COLUMN = 'status',
                  CONSTRAINT = 'profile_revision_status_transition',
                  HINT = '허용 전이: DRAFT→TESTING, TESTING→PUBLISHED, PUBLISHED→RETIRED, '
                         'TESTING→RETIRED, RETIRED→PUBLISHED';
    END IF;
    RETURN NEW;
END;
""",
    "chat_profile_revision_lock_delete": r"""
BEGIN
    IF OLD.status <> 'DRAFT' THEN
        RAISE EXCEPTION
            'DRAFT 프로필 버전만 지울 수 있습니다: chat_profile_revisions.id=%, status=%',
            OLD.id, OLD.status
            USING ERRCODE = 'restrict_violation',
                  TABLE = TG_TABLE_NAME,
                  COLUMN = 'status',
                  CONSTRAINT = 'profile_revision_delete_draft_only',
                  HINT = '내리려면 RETIRED 로 바꿉니다.';
    END IF;
    IF EXISTS (SELECT 1 FROM conversations WHERE chat_profile_revision_id = OLD.id) THEN
        RAISE EXCEPTION
            '서빙된 프로필 버전은 지울 수 없습니다(conversations.chat_profile_revision_id 가 가리킴): '
            'chat_profile_revisions.id=%', OLD.id
            USING ERRCODE = 'restrict_violation',
                  TABLE = TG_TABLE_NAME,
                  CONSTRAINT = 'profile_revision_delete_never_served';
    END IF;
    IF EXISTS (SELECT 1 FROM rag_runs WHERE profile_revision_id = OLD.id) THEN
        RAISE EXCEPTION
            '서빙된 프로필 버전은 지울 수 없습니다(rag_runs.profile_revision_id 가 가리킴): '
            'chat_profile_revisions.id=%', OLD.id
            USING ERRCODE = 'restrict_violation',
                  TABLE = TG_TABLE_NAME,
                  CONSTRAINT = 'profile_revision_delete_never_served';
    END IF;
    RETURN OLD;
END;
""",
    "chat_profile_revision_note_insert": rf"""
BEGIN
    PERFORM set_config(
        '{CREATED_IDS_SETTING}',
        concat_ws(',', nullif(current_setting('{CREATED_IDS_SETTING}', true), ''), NEW.id),
        true
    );
    RETURN NULL;
END;
""",
    "profile_revision_component_lock": rf"""
DECLARE
    target_id bigint;
    parent_status text;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION
            '프로필 버전 구성 행은 바꿀 수 없습니다: profile_revision_components.id=%', OLD.id
            USING ERRCODE = 'restrict_violation',
                  TABLE = TG_TABLE_NAME,
                  CONSTRAINT = 'profile_revision_component_immutable',
                  HINT = '고치려면 새 프로필 버전을 만듭니다.';
    END IF;

    IF TG_OP = 'DELETE' THEN
        target_id := OLD.revision_id;
    ELSE
        target_id := NEW.revision_id;
    END IF;
    SELECT status INTO parent_status
    FROM chat_profile_revisions WHERE id = target_id FOR SHARE;

    IF TG_OP = 'DELETE' THEN
        IF parent_status IS DISTINCT FROM 'DRAFT' THEN
            RAISE EXCEPTION
                'DRAFT 프로필 버전의 구성 행만 지울 수 있습니다: profile_revision_components.id=%, '
                'chat_profile_revisions.id=%, status=%', OLD.id, target_id, parent_status
                USING ERRCODE = 'restrict_violation',
                      TABLE = TG_TABLE_NAME,
                      CONSTRAINT = 'profile_revision_component_delete_draft_only';
        END IF;
        RETURN OLD;
    END IF;

    -- 부모가 없으면 외래키가 거부한다.
    IF parent_status IS NULL OR parent_status = 'DRAFT' OR CAST(target_id AS text) = ANY(
        string_to_array(current_setting('{CREATED_IDS_SETTING}', true), ',')
    ) THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION
        '구성 행은 DRAFT 프로필 버전이나 같은 트랜잭션에서 만든 프로필 버전에만 넣을 수 있습니다: '
        'chat_profile_revisions.id=%, status=%', target_id, parent_status
        USING ERRCODE = 'restrict_violation',
              TABLE = TG_TABLE_NAME,
              CONSTRAINT = 'profile_revision_component_insert_draft_only';
END;
""",
}


def upgrade() -> None:
    for name, body in FUNCTIONS.items():
        op.execute(
            f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $body${body}$body$"
        )
    for trigger, table, timing, function in TRIGGERS:
        op.execute(
            f"CREATE TRIGGER {trigger} {timing} ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION {function}()"
        )


def downgrade() -> None:
    for trigger, table, _, _ in reversed(TRIGGERS):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
    for name in reversed(list(FUNCTIONS)):
        op.execute(f"DROP FUNCTION IF EXISTS {name}()")
