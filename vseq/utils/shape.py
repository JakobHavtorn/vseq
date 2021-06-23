from typing import List, Tuple


def concatenate_shapes(shapes: List[Tuple[int]], axis: int):
    """Concatenate shapes along axis"""
    out = list(shapes[0])
    out[axis] = sum(list(s)[axis] for s in shapes)
    return tuple(out)
