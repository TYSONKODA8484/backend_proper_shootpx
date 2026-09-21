from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# pool_recycle: refresh connections before the server would cut them.
# If you switch to Supabase's transaction pooler (port 6543), also pass
# poolclass=NullPool and connect_args={"prepare_threshold": None}.
#
# pool_pre_ping is deliberately OFF. It issues a `SELECT 1` on EVERY connection
# checkout -- measured at ~145-200ms here, because the database is a full
# network round trip away, so it was adding a whole wasted round trip to every
# single request. pool_recycle above already retires connections well before
# the server or pooler would drop them, which covers the staleness pre_ping
# exists to catch. The residual risk is a connection that dies in the window
# between its last use and recycle: that surfaces as one clean, retryable
# connection error instead of being silently absorbed. That is the standard
# trade-off and it is worth it at this latency -- re-enable pre_ping only if
# dead-connection errors actually start appearing in logs.
#
# connect_args: the database is reached over the internet (Supabase's pooler),
# and psycopg2 has NO default timeout -- if a connection silently dies
# mid-query (wifi change, laptop sleep, NAT/pooler dropping an idle socket),
# the client waits on it FOREVER. In the arq worker that is fatal: every job
# and cron shares one event loop, so a single hung query froze the whole
# worker (confirmed with a stack dump: MainThread stuck in psycopg2's execute
# inside check_generation_timeouts) while the process stayed "running" and
# every queued generation sat there untouched. TCP keepalives make the OS
# detect a dead peer (~30s idle + 3 probes x 10s = about 60s) and raise a
# normal connection error instead, which arq logs and moves past.
# connect_timeout bounds a hang while opening a connection.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=False,
    pool_recycle=1800,
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
    },
)

# autocommit=False is REQUIRED, not an oversight -- several paths depend on
# multi-statement atomicity and would be corrupted by per-statement commits:
#   * create_generation_batch (services/generation.py): deducts credits and
#     inserts the jobs in ONE transaction, so a failure can never leave a team
#     charged with no jobs to show for it.
#   * handle_payment_captured (services/webhooks.py): claims the Razorpay
#     payment id and grants the credits together -- that shared transaction IS
#     the idempotency guard against a retried webhook double-crediting.
#   * _create_user_with_team (services/users.py): user + team + membership.
# A blanket AUTOCOMMIT engine would remove the ~290ms BEGIN/ROLLBACK pair per
# request but break all three. See the note in get_db() for why a separate
# read-only autocommit session isn't a free win either.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()