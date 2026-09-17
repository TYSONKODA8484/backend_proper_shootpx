"""app/services/credits.py::spend_credits_up_to -- the partial-fulfillment
primitive create_generation_batch uses for multi-output batches (see
tests/test_generation.py's listing_photoshoot pricing section for the
end-to-end scenario)."""

import uuid
from unittest.mock import MagicMock

import pytest

from app.services.credits import spend_credits_up_to

TEAM_ID = uuid.uuid4()


def _db_with_team(subscription_remaining: int, topup_balance: int):
    team = MagicMock(subscription_credits_remaining=subscription_remaining, topup_credits_balance=topup_balance)
    db = MagicMock()
    db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = team
    return db, team


def test_grants_the_full_request_when_fully_affordable():
    db, team = _db_with_team(subscription_remaining=20, topup_balance=0)

    _, granted, from_sub, from_top = spend_credits_up_to(db, TEAM_ID, 2, 6, commit=False)

    assert granted == 6
    assert from_sub == 12
    assert from_top == 0
    assert team.subscription_credits_remaining == 8


def test_grants_a_smaller_batch_when_only_partially_affordable():
    """The exact scenario reported live: 6 units at 2 credits each (12
    needed) against a team with only 10 truly available -- grants 5 whole
    units (10), not 6, and never rejects the request outright."""
    db, team = _db_with_team(subscription_remaining=10, topup_balance=0)

    _, granted, from_sub, from_top = spend_credits_up_to(db, TEAM_ID, 2, 6, commit=False)

    assert granted == 5
    assert from_sub == 10
    assert from_top == 0


def test_raises_when_not_even_one_unit_is_affordable():
    db, team = _db_with_team(subscription_remaining=1, topup_balance=0)

    with pytest.raises(ValueError, match="Insufficient credits"):
        spend_credits_up_to(db, TEAM_ID, 2, 6, commit=False)


def test_single_unit_request_behaves_like_plain_spend_credits():
    db, team = _db_with_team(subscription_remaining=5, topup_balance=0)

    _, granted, from_sub, from_top = spend_credits_up_to(db, TEAM_ID, 5, 1, commit=False)

    assert granted == 1
    assert from_sub == 5
    assert from_top == 0


def test_zero_cost_grants_the_full_request_without_dividing_by_zero():
    """Regression: a genuinely free tool (per_unit_cost <= 0, e.g. a flat
    credit_cost_per_output of 0) must grant the full request regardless of
    balance -- `total_available // per_unit_cost` previously crashed with
    ZeroDivisionError instead."""
    db, team = _db_with_team(subscription_remaining=0, topup_balance=0)

    _, granted, from_sub, from_top = spend_credits_up_to(db, TEAM_ID, 0, 3, commit=False)

    assert granted == 3
    assert from_sub == 0
    assert from_top == 0


def test_splits_across_subscription_and_topup_pools_for_a_partial_grant():
    db, team = _db_with_team(subscription_remaining=3, topup_balance=7)

    # 5 units at 2 credits = 10 needed, 10 available -- full grant, split
    # subscription-first then topup, same convention as spend_credits.
    _, granted, from_sub, from_top = spend_credits_up_to(db, TEAM_ID, 2, 5, commit=False)

    assert granted == 5
    assert from_sub == 3
    assert from_top == 7


def test_raises_when_team_not_found():
    db = MagicMock()
    db.query.return_value.filter.return_value.with_for_update.return_value.first.return_value = None

    with pytest.raises(ValueError, match="Team not found"):
        spend_credits_up_to(db, TEAM_ID, 2, 3, commit=False)


def test_commit_false_does_not_call_db_commit():
    db, team = _db_with_team(subscription_remaining=10, topup_balance=0)

    spend_credits_up_to(db, TEAM_ID, 2, 3, commit=False)

    db.commit.assert_not_called()
