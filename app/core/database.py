from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# pool_pre_ping: drop dead connections (Supabase/poolers close idle ones).
# pool_recycle: refresh connections before the server would cut them.
# If you switch to Supabase's transaction pooler (port 6543), also pass
# poolclass=NullPool and connect_args={"prepare_threshold": None}.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()