"""Object-pair grids for state-based linear probes."""

from __future__ import annotations

from typing import Sequence

# (obj1, obj2): goal +10, lure -10. Labels decorrelated across train grid.
STATE_PROBE_TRAIN_CONFIGS = [
    ("yellow_line_diag", "red_gem"),
    ("red_line_diag", "blue_gem"),
    ("yellow_gem", "red_gem"),
    ("blue_gem", "red_gem"),
    ("green_line_diag", "yellow_gem"),
    ("green_line_diag", "blue_gem"),
    ("green_gem", "yellow_gem"),
    ("green_gem", "blue_gem"),
]

# Paint × partner factorial for OOD-transport Canon (published agents).
# Paints = green, red (yellow held out — it is the proxy colour).
# Partners = blue / yellow / purple gems (colour-disjoint from paints).
# Every (paint, partner) cell exists → partner axis can hold paint fixed.
# Negatives = line-absent gem–gem pairs for partition balance.
STATE_PROBE_FACTORIAL_TRAIN_CONFIGS = [
    ("green_line_diag", "blue_gem"),
    ("green_line_diag", "yellow_gem"),
    ("green_line_diag", "purple_gem"),
    ("red_line_diag", "blue_gem"),
    ("red_line_diag", "yellow_gem"),
    ("red_line_diag", "purple_gem"),
    ("blue_gem", "yellow_gem"),
    ("blue_gem", "purple_gem"),
    ("yellow_gem", "purple_gem"),
    ("green_gem", "blue_gem"),
    ("green_gem", "yellow_gem"),
    ("green_gem", "purple_gem"),
]

STATE_PROBE_OOD_CONFIGS = [
    ("red_line_diag", "yellow_gem"),
    ("white_line_diag", "blue_gem"),
    ("purple_gem", "yellow_gem"),
    ("purple_gem", "blue_gem"),
]

# Index in STATE_PROBE_OOD_CONFIGS — classic GM trap (shape goal, colour lure).
STATE_PROBE_GM_TRAP_OOD_INDEX = 0

STATE_PROBE_CONCEPTS = {
    # Presence, not goal-role. On the published grid lines never appear as
    # lures, so this coincides with goal_is_line; they diverge on lure-line grids.
    "line_present": "shape presence (a line-shaped object is present in either slot)",
    "yellow_present": "proxy colour presence (a yellow object is present in either slot)",
}

PUBLISHED_PROBE_SEEDS = ("683", "3561", "7257", "3959", "8038")

# ---------------------------------------------------------------------------
# Goal-identity four-way discriminating grid
# ---------------------------------------------------------------------------
# Shape: line-as-goal / line-as-lure / line-absent so presence ≠ goal-role.
# Colour: OOD includes yellow as +10 so goal_is_yellow is non-degenerate.

GOAL_IDENTITY_GRID_TAG = "discriminating_four_way_v2"

GOAL_IDENTITY_CONCEPTS = (
    "line_present",
    "goal_is_line",
    "yellow_present",
    "goal_is_yellow",
)

GOAL_IDENTITY_TRAIN_CONFIGS: list[tuple[str, str]] = [
    # A: line goal, gem lure  (Lp=1, Gl=1)
    ("red_line_diag", "blue_gem"),
    ("green_line_diag", "purple_gem"),
    ("red_line_diag", "yellow_gem"),
    ("green_line_diag", "blue_gem"),
    # B: gem goal, line lure  (Lp=1, Gl=0)
    ("blue_gem", "red_line_diag"),
    ("purple_gem", "green_line_diag"),
    ("yellow_gem", "red_line_diag"),
    ("blue_gem", "green_line_diag"),
    # C: gem–gem             (Lp=0, Gl=0)
    ("blue_gem", "red_gem"),
    ("green_gem", "purple_gem"),
    ("yellow_gem", "blue_gem"),
    ("purple_gem", "red_gem"),
]

GOAL_IDENTITY_OOD_CONFIGS: list[tuple[str, str]] = [
    ("white_line_diag", "cyan_gem"),   # A held-out
    ("cyan_gem", "white_line_diag"),   # B held-out
    ("white_gem", "cyan_gem"),         # C held-out
    ("cyan_line_diag", "yellow_gem"),  # Y lure (Yp=1, Gy=0)
    ("yellow_gem", "cyan_line_diag"),  # Y goal + line lure (Gy=1, Gl=0)
    ("yellow_line_diag", "cyan_gem"),  # Y goal + line goal (Gy=1, Gl=1)
]


def audit_goal_identity_grid(
    configs: Sequence[tuple[str, str]] | None = None,
    *,
    assert_gates: bool = True,
    split_name: str = "configs",
) -> dict:
    """Label rates / correlations for a goal-identity duel mix.

    Hard gate (when ``assert_gates``): each of ``GOAL_IDENTITY_CONCEPTS`` has
    both classes (``0 < rate < 1``). Import of ``ground_truth_presence`` is
    deferred to avoid a circular import through ``probes.__init__``.
    """
    from gm_geometry.probes.labels import ground_truth_presence

    cfgs = list(configs) if configs is not None else list(GOAL_IDENTITY_TRAIN_CONFIGS)
    rows = [ground_truth_presence(a, b) for a, b in cfgs]
    out: dict = {"n_configs": len(rows), "split": split_name, "rates": {}, "correlations": {}}
    series: dict[str, list[float]] = {k: [] for k in GOAL_IDENTITY_CONCEPTS}
    for r in rows:
        for k in GOAL_IDENTITY_CONCEPTS:
            series[k].append(float(r[k]))

    for k, vals in series.items():
        rate = sum(vals) / len(vals) if vals else float("nan")
        out["rates"][k] = rate
        if assert_gates and not (0.0 < rate < 1.0):
            raise AssertionError(
                f"goal-identity {split_name}: label {k!r} degenerate "
                f"(rate={rate:.3f}, n={len(vals)}). Need both classes."
            )

    def _corr(a: str, b: str) -> float:
        xs, ys = series[a], series[b]
        n = len(xs)
        if n < 2:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
        den_y = sum((y - my) ** 2 for y in ys) ** 0.5
        if den_x == 0.0 or den_y == 0.0:
            return float("nan")
        return float(num / (den_x * den_y))

    out["correlations"] = {
        "corr_Lp_Gl": _corr("line_present", "goal_is_line"),
        "corr_Yp_Gy": _corr("yellow_present", "goal_is_yellow"),
        "corr_Lp_Yp": _corr("line_present", "yellow_present"),
        "corr_Gl_Gy": _corr("goal_is_line", "goal_is_yellow"),
    }
    out["n_disagree_Lp_vs_Gl"] = int(
        sum(1 for a, b in zip(series["line_present"], series["goal_is_line"]) if a != b)
    )
    out["n_disagree_Yp_vs_Gy"] = int(
        sum(1 for a, b in zip(series["yellow_present"], series["goal_is_yellow"]) if a != b)
    )
    return out


def audit_goal_identity_grids(*, assert_gates: bool = True) -> dict:
    """Audit train + OOD; optionally enforce both-class gates on all four labels."""
    return {
        "grid": GOAL_IDENTITY_GRID_TAG,
        "train": audit_goal_identity_grid(
            GOAL_IDENTITY_TRAIN_CONFIGS, assert_gates=assert_gates, split_name="train",
        ),
        "ood": audit_goal_identity_grid(
            GOAL_IDENTITY_OOD_CONFIGS, assert_gates=assert_gates, split_name="ood",
        ),
    }


# ---------------------------------------------------------------------------
# Matched-difficulty role reversal probe grids
# ---------------------------------------------------------------------------
# Primary: colour–colour on fixed shape (gem). Concepts = ``{c}_present``.
# Fillers (green/yellow/purple/cyan/white) vary presence of the pair colours
# without putting the held-out OOD pairs in the train grid.

REVERSAL_COLOUR_PAIRS = (
    ("red", "blue"),    # primary RGB
    ("red", "green"),
    ("green", "blue"),
)

# Shape inventory in maze_colorsobjects_duel: only gem + line_diag.
# Coloured pixel mass (black-bg pure assets): gem ≈ 25, line_diag ≈ 36 — not matched.
REVERSAL_SHAPE_PAIRS = (
    ("gem", "line_diag"),
)


def colour_colour_probe_grids(
    colour_a: str,
    colour_b: str,
    *,
    shape: str = "gem",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], tuple[str, str]]:
    """Train/OOD duel grids so presence of ``colour_a`` and ``colour_b`` decorrelate.

    Returns ``(train_configs, ood_configs, concept_keys)``.
    """
    a, b = colour_a, colour_b
    sa, sb = f"{a}_{shape}", f"{b}_{shape}"
    # Fillers: not A or B.
    fillers = [c for c in ("green", "yellow", "purple", "cyan", "white") if c not in (a, b)]
    f0, f1, f2, f3 = fillers[0], fillers[1], fillers[2], fillers[3]

    train = [
        (sa, f"{f0}_{shape}"),           # A=1 B=0
        (sa, f"{f1}_{shape}"),           # A=1 B=0
        (sb, f"{f0}_{shape}"),           # A=0 B=1
        (sb, f"{f1}_{shape}"),           # A=0 B=1
        (sa, sb),                        # A=1 B=1
        (f"{f0}_{shape}", f"{f1}_{shape}"),  # A=0 B=0
        (f"{f2}_{shape}", f"{f3}_{shape}"),  # A=0 B=0
        (f"{f0}_{shape}", f"{f2}_{shape}"),  # A=0 B=0
    ]
    ood = [
        (sa, f"{f2}_{shape}"),           # A=1 B=0 held-out filler
        (sb, f"{f2}_{shape}"),           # A=0 B=1
        (sa, f"{f3}_{shape}"),           # A=1 B=0
        (f"{f1}_{shape}", f"{f2}_{shape}"),  # A=0 B=0
    ]
    concepts = (f"{a}_present", f"{b}_present")
    return train, ood, concepts


def shape_shape_probe_grids(
    shape_a: str,
    shape_b: str,
    *,
    colour: str = "red",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], tuple[str, str]]:
    """Train/OOD grids for two shapes on a fixed colour (fallback; not pixel-matched).

    Concepts: ``line_present`` / ``gem_present`` when shapes are line_diag/gem.
    """
    # Only gem ↔ line_diag exist in the fork.
    if {shape_a, shape_b} != {"gem", "line_diag"}:
        raise ValueError(f"Unsupported shape pair: {shape_a}, {shape_b}")

    def _obj(shape: str, col: str) -> str:
        return f"{col}_line_diag" if shape == "line_diag" else f"{col}_gem"

    # Vary colour so line/gem presence can decorrelate from any single hue.
    colours_train = ("red", "blue", "green", "yellow")
    colours_ood = ("purple", "cyan", "white", "green")

    train: list[tuple[str, str]] = []
    # Dual-shape (both present): line vs gem, several colours
    for c1, c2 in (("red", "blue"), ("green", "yellow"), ("blue", "green"), ("yellow", "red")):
        train.append((_obj("line_diag", c1), _obj("gem", c2)))
    # Line-only (gem absent): two lines
    train.append((_obj("line_diag", "red"), _obj("line_diag", "blue")))
    train.append((_obj("line_diag", "green"), _obj("line_diag", "yellow")))
    # Gem-only (line absent): two gems
    train.append((_obj("gem", "red"), _obj("gem", "blue")))
    train.append((_obj("gem", "green"), _obj("gem", "yellow")))

    ood = [
        (_obj("line_diag", "purple"), _obj("gem", "cyan")),
        (_obj("line_diag", "white"), _obj("gem", "purple")),
        (_obj("line_diag", "cyan"), _obj("line_diag", "purple")),
        (_obj("gem", "white"), _obj("gem", "cyan")),
    ]
    _ = (colour, colours_train, colours_ood)  # reserved for future colour-locked variant
    return train, ood, ("line_present", "gem_present")
