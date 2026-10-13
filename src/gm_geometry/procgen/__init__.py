"""Deprecated alias of ``gm_geometry.integrations.colour_shape`` plus VecEnv wrapper-unwrapping helpers."""

from __future__ import annotations

from typing import Iterable

from gm_geometry.integrations.colour_shape import *  # noqa: F403
from gm_geometry.integrations.colour_shape import (
    collect_activations,
    ensure_imports,
    load_impala_policy,
    load_recurrent_flag,
    make_duel_env,
    make_procgen_env,
    repo_root,
    resolve_train_root,
)

ensure_procgen_imports = ensure_imports

_DEFAULT_CHAIN = ("env", "venv", "_env")


def _step_wrapper(cur, chain: Iterable[str]):
    for attr in chain:
        nxt = getattr(cur, attr, None)
        if nxt is not None:
            return nxt
    return None


def unwrap_base(
    env,
    required_attr: str | Iterable[str],
    *,
    chain: Iterable[str] = _DEFAULT_CHAIN,
    on_missing: str = "raise",
):
    """Walk a VecEnv wrapper stack (``env``/``venv``/``_env``) for the env exposing ``required_attr``.

    ``required_attr`` is a method name or an iterable of names that must all be present. ``on_missing`` is
    ``"raise"`` (RuntimeError) or ``"none"`` (return None)."""
    attrs = (required_attr,) if isinstance(required_attr, str) else tuple(required_attr)
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if all(hasattr(cur, a) and callable(getattr(cur, a)) for a in attrs):
            return cur
        cur = _step_wrapper(cur, chain)
    if on_missing == "none":
        return None
    raise RuntimeError(f"could not unwrap env exposing {attrs!r}")


def unwrap_preferred(
    env,
    primary_attr: str,
    prefer_attrs: str | Iterable[str],
    *,
    chain: Iterable[str] = _DEFAULT_CHAIN,
    on_missing: str = "raise",
):
    """Env exposing ``primary_attr``, preferring the first one along the chain that also has every attr in ``prefer_attrs``."""
    prefer = (prefer_attrs,) if isinstance(prefer_attrs, str) else tuple(prefer_attrs)
    candidates = []
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, primary_attr) and callable(getattr(cur, primary_attr)):
            candidates.append(cur)
        cur = _step_wrapper(cur, chain)
    for c in candidates:
        if all(hasattr(c, a) for a in prefer):
            return c
    if candidates:
        return candidates[0]
    if on_missing == "none":
        return None
    raise RuntimeError(f"could not unwrap env exposing {primary_attr!r}")


def get_info(env, *, chain: Iterable[str] = ("venv", "env"), all_infos: bool = False):
    """Call ``.get_info()`` on the wrapper exposing it (gym3-style list of per-env dicts, or a bare dict).

    Returns the first env's info dict, or the full list with ``all_infos=True``."""
    base = unwrap_base(env, "get_info", chain=chain)
    infos = base.get_info()
    if all_infos:
        if isinstance(infos, dict):
            return [infos]
        return [dict(x) if isinstance(x, dict) else {} for x in infos]
    if isinstance(infos, dict):
        return infos
    return dict(infos[0]) if infos else {}
