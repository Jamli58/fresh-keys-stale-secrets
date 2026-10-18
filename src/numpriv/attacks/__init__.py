"""Attackers see only an AttackerView; ground truth (truths_for) is used for scoring only.

M0 provides A0 (prior-only floor) and A1's plaintext branch (D0 sanity).
The classical Bayesian attacker for disguised inputs (A1/A1+) is milestone M1.
"""

from numpriv.attacks.baselines import PlaintextAttack, PriorOnlyAttack
from numpriv.attacks.view import AttackerView, Estimate, Oracle, make_view, truths_for

__all__ = ["AttackerView", "Estimate", "Oracle", "PlaintextAttack", "PriorOnlyAttack", "make_view", "truths_for"]
