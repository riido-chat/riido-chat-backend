"""애플리케이션 ORM model이 공유하는 SQLAlchemy Base를 정의한다."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """모든 ORM model이 상속하는 공용 declarative base."""
