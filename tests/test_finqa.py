import pytest

from numpriv.finqa import (
    answers_match,
    execute_program,
    is_arithmetic_only,
    parse_program,
    program_literals,
    substitute_literals,
)


def test_parse_simple_program():
    steps = parse_program("subtract(5829, 5735), divide(#0, 5735)")
    assert steps == (("subtract", ("5829", "5735")), ("divide", ("#0", "5735")))


def test_execute_arithmetic_with_references_and_constants():
    assert execute_program("subtract(5829, 5735)") == pytest.approx(94.0)
    assert execute_program("subtract(5829, 5735), divide(#0, 5735)") == pytest.approx(94 / 5735)
    assert execute_program("divide(3, const_100)") == pytest.approx(0.03)
    assert execute_program("multiply(2, const_m1)") == pytest.approx(-2.0)
    assert execute_program("exp(2, 3)") == pytest.approx(8.0)


def test_execute_percent_literal_follows_finqa_convention():
    assert execute_program("multiply(200, 35%)") == pytest.approx(70.0)


def test_execute_greater_returns_yes_no():
    assert execute_program("greater(5, 3)") == "yes"
    assert execute_program("greater(2, 3)") == "no"


def test_table_ops_are_flagged_not_executed():
    assert not is_arithmetic_only("table_max(net revenue, none)")
    assert is_arithmetic_only("add(1, 2), divide(#0, const_2)")
    with pytest.raises(ValueError):
        execute_program("table_max(net revenue, none)")


def test_program_literals_exclude_constants_and_references():
    lits = program_literals("subtract(5829, 5735), divide(#0, const_100), multiply(#1, 35%)")
    assert lits == ("5829", "5735", "35%")


def test_substitute_literals_rewrites_only_mapped_literals():
    prog = "subtract(5829, 5735), divide(#0, 5735)"
    out = substitute_literals(prog, {"5829": "10.5", "5735": "3.25"})
    assert out == "subtract(10.5, 3.25), divide(#0, 3.25)"


def test_answers_match_numeric_and_boolean():
    assert answers_match(0.0163905, 94 / 5735)
    assert not answers_match(0.02, 94 / 5735)
    assert answers_match("yes", "yes")
    assert not answers_match("yes", 1.0)


def test_answers_match_follows_finqa_five_decimal_convention():
    """FinQA stores exe_ans rounded to 5 decimals and its evaluator compares at that precision."""
    assert answers_match(0.017144688351225765, 0.01714)
    assert answers_match(7.474549160109826e-06, 1e-05)
    assert not answers_match(0.01716, 0.01714)
