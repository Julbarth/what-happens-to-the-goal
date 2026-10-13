"""Read the agent's grid cell from a procgen get_state() buffer, without rendering or stepping."""
# Entity::serialize writes x, y as each entity's first two float32s and grid_step games keep the agent at
# cell centres (x = cell + 0.5); the byte offset is fixed within an episode, so it is located once and reused.
from __future__ import annotations

import math
import struct


def raw_state(base_api) -> bytes:
    st = base_api.get_state()
    return bytes(st[0]) if isinstance(st, (list, tuple)) else bytes(st)


def find_agent_xy_offsets(raw: bytes, cell: tuple[int, int]) -> list[int]:
    """All byte offsets holding float32 (cell_x + 0.5, cell_y + 0.5)."""
    target = struct.pack("<ff", cell[0] + 0.5, cell[1] + 0.5)
    out, i = [], raw.find(target)
    while i != -1:
        out.append(i)
        i = raw.find(target, i + 1)
    return out


def read_agent_cell(raw: bytes, offset: int) -> tuple[int, int]:
    x, y = struct.unpack_from("<ff", raw, offset)
    return int(math.floor(x)), int(math.floor(y))
