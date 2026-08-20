"""애플리케이션 ORM model이 공유하는 SQLAlchemy Base를 정의한다."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# 제약조건 이름을 결정적으로 생성해 Alembic migration과 ORM 정의를 일치시킨다.
# 기존 수동 명명 규칙(pk_*, fk_*_*, uq_*)과 동일한 형식을 사용한다.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """모든 ORM model이 상속하는 공용 declarative base."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
