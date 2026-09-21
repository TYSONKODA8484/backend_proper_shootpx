"""Isolate: with pre_ping already off, how much MORE would AUTOCOMMIT save
on top of that, for a 2-query read-only session? Measures both configs
back to back against the same live DB so the comparison is apples-to-apples.
"""
import os, time, statistics
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

URL = os.environ["DATABASE_URL"]


def bench(label, isolation_level=None):
    eng = create_engine(URL, pool_pre_ping=False, pool_recycle=1800,
                        isolation_level=isolation_level) if isolation_level \
        else create_engine(URL, pool_pre_ping=False, pool_recycle=1800)
    S = sessionmaker(bind=eng)
    s = S(); s.execute(text("SELECT 1")); s.close()  # warm

    totals = []
    for _ in range(5):
        t0 = time.perf_counter()
        db = S()
        db.connection()
        db.execute(text("SELECT 1")).scalar()
        db.execute(text("SELECT 1")).scalar()
        db.close()
        totals.append((time.perf_counter() - t0) * 1000)
    eng.dispose()
    print(f"  {label:42} median {statistics.median(totals):7.1f} ms   samples={[f'{t:.0f}' for t in totals]}")
    return statistics.median(totals)


print("=== incremental effect of AUTOCOMMIT on top of pre_ping=False ===")
default_tx = bench("pre_ping=False, default (BEGIN/ROLLBACK)")
autocommit = bench("pre_ping=False, isolation_level=AUTOCOMMIT", isolation_level="AUTOCOMMIT")
print(f"\n  additional saving from AUTOCOMMIT: {default_tx - autocommit:.1f} ms per 2-query request")
