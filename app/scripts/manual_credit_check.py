"""Manual smoke test for the credit helpers against the real database.

NOT a pytest test — it mutates real rows. Run it by hand:

    python -m app.scripts.manual_credit_check <team_id>

(named without a `test_` prefix and guarded by __main__ so `pytest` never
collects or executes it.)
"""

import sys

from app.core.database import SessionLocal
from app.models.team import Team
from app.services.credits import add_topup_credits, get_total_credits, spend_credits


def main(team_id: str) -> None:
    db = SessionLocal()
    try:
        print("--- Before ---")
        team = db.query(Team).filter(Team.id == team_id).first()
        if not team:
            print(f"no team with id {team_id!r}")
            return
        print(f"subscription_credits_remaining: {team.subscription_credits_remaining}")
        print(f"topup_credits_balance: {team.topup_credits_balance}")
        print(f"total: {get_total_credits(team)}")

        print("\n--- Adding 100 topup credits ---")
        team = add_topup_credits(db, team_id, 100)
        print(f"topup_credits_balance: {team.topup_credits_balance}")

        print("\n--- Spending 30 credits ---")
        team = spend_credits(db, team_id, 30)
        print(f"subscription_credits_remaining: {team.subscription_credits_remaining}")
        print(f"topup_credits_balance: {team.topup_credits_balance}")
        print(f"total: {get_total_credits(team)}")

        print("\n--- Trying to spend more than available ---")
        try:
            spend_credits(db, team_id, 999999)
        except ValueError as e:
            print(f"Correctly rejected: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.scripts.manual_credit_check <team_id>")
    main(sys.argv[1])
