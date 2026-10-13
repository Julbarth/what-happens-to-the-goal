"""Dependency-free cell-geometry predicates for the canonical Coin bank's policy-blind task-conflict gate."""
# Cells follow coin_location.find_floor_cell: integer columns measured from int(agent_x), so the gate
# compares cells to cells, never cells to the continuous coordinate.
from __future__ import annotations

MIN_DX = 1   # cells the goal/shortcut coin must sit left/right of the agent cell


def agent_cell_x(agent_x: float) -> int:
    """The agent's grid column, using int() exactly as find_floor_cell does."""
    return int(agent_x)


def targets_in_opposition(agent_x: float, goal_x: int, shortcut_x: int,
                          min_dx: int = MIN_DX) -> bool:
    """True when the goal cell is at least ``min_dx`` columns left of the agent and the shortcut cell at least
    ``min_dx`` columns right, so the task-consistent actions differ for every policy."""
    ax = agent_cell_x(agent_x)
    return (goal_x - ax) <= -min_dx and (shortcut_x - ax) >= min_dx
