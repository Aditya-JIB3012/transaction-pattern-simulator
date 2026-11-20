"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Represent anonymized financial transactions as ordered event sequences.
def test_milestone_1():
    from app.core import milestone_1, transactions_to_sequences, FinancialTransaction

    # 1. Test default benchmark generation
    sequences = milestone_1()
    assert isinstance(sequences, list)
    assert len(sequences) == 200
    assert all(isinstance(seq, list) for seq in sequences)
    assert all(len(seq) >= 4 for seq in sequences)
    assert all(isinstance(event, str) for seq in sequences for event in seq)

    # 2. Test structuring custom raw transactions into chronological sequences
    custom_txs = [
        {"customer_id": "cust_1", "timestamp": "2026-01-01T10:00:00", "event_type": "Salary_Direct_Deposit"},
        {"customer_id": "cust_2", "timestamp": "2026-01-01T09:00:00", "event_type": "Grocery_POS"},
        {"customer_id": "cust_1", "timestamp": "2026-01-02T12:00:00", "event_type": "Rent_Mortgage_Payment"},
        {"customer_id": "cust_1", "timestamp": "2026-01-01T11:00:00", "event_type": "Utility_Bill_Payment"},
    ]
    seqs = transactions_to_sequences(custom_txs)
    assert len(seqs) == 2
    # cust_1 should have chronological order
    assert seqs[0] == ["Salary_Direct_Deposit", "Utility_Bill_Payment", "Rent_Mortgage_Payment"]
    assert seqs[1] == ["Grocery_POS"]

    # 3. Test with FinancialTransaction objects
    obj_txs = [
        FinancialTransaction(customer_id="c3", timestamp="2026-01-03T10:00:00", event_type="ATM_Cash_Withdrawal"),
        FinancialTransaction(customer_id="c3", timestamp="2026-01-03T11:00:00", event_type="Pharmacy_Medical"),
    ]
    obj_seqs = milestone_1(obj_txs)
    assert obj_seqs == [["ATM_Cash_Withdrawal", "Pharmacy_Medical"]]


# Milestone 2: Discover frequent transaction patterns using `thilimarketbasket`.
def test_milestone_2():
    from app.core import milestone_2, discover_transaction_patterns

    # 1. Test mining on default transaction sequences
    result = milestone_2(min_support=0.08, min_confidence=0.3)
    assert isinstance(result, dict)
    assert "patterns" in result
    assert "rules" in result
    assert "summary" in result
    assert result["summary"]["total_sequences"] == 200
    assert result["summary"]["num_patterns"] > 0
    assert result["summary"]["num_rules"] > 0

    # 2. Test mining on controlled sequences
    controlled_sequences = [
        ["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Utility_Bill_Payment"],
        ["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Grocery_POS"],
        ["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Utility_Bill_Payment"],
        ["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Gas_Station_POS"],
    ]
    res = discover_transaction_patterns(controlled_sequences, min_support=0.5, min_confidence=0.5)
    # Salary and Rent appear in all 4 (support 1.0)
    assert frozenset(["Salary_Direct_Deposit", "Rent_Mortgage_Payment"]) in res["patterns"]
    assert res["patterns"][frozenset(["Salary_Direct_Deposit", "Rent_Mortgage_Payment"])] == 1.0
    # Utility appears in 2 out of 4 (support 0.5)
    assert frozenset(["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Utility_Bill_Payment"]) in res["patterns"]
    assert res["patterns"][frozenset(["Salary_Direct_Deposit", "Rent_Mortgage_Payment", "Utility_Bill_Payment"])] == 0.5


# Milestone 3: Fine-tune an LM on these sequences to generate new, plausible transaction streams.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate the generated sequences for similarity to real patterns.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a CLI to generate sequences based on a starting set of transactions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

