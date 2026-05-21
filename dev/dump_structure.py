from typing import Any, Optional, TextIO
import sys

def dump_structure(obj: Any, name: str = "root", prefix: str = "", is_last: bool = True, 
                   max_depth: int = 3, current_depth: int = 0, file: TextIO = sys.stdout):
    if current_depth > max_depth:
        return

    # 1. Visual Formatting
    branch = "" if current_depth == 0 else ("└── " if is_last else "├── ")
    type_name = type(obj).__name__
    file.write(f"{prefix}{branch}{name} ({type_name})\n")

    # 2. Prepare Prefix for Children
    if current_depth == 0:
        new_prefix = "" 
    else:
        new_prefix = prefix + ("    " if is_last else "│   ")

    # --- 核心修正：定义基础类型，不再向下递归 ---
    ATOMIC_TYPES = (int, float, complex, bool, str, bytes, type(None))
    if isinstance(obj, ATOMIC_TYPES):
        return
    # ---------------------------------------

    # 3. Extract Children
    children = {}
    if isinstance(obj, dict):
        children = obj
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) > 0:
            children = {"[element]": obj[0]}
        else:
            return
    else:
        try:
            for attr in dir(obj):
                if not attr.startswith('_'):
                    val = getattr(obj, attr)
                    if not callable(val):
                        children[attr] = val
        except Exception:
            return

    # 4. Recurse through Children
    items = list(children.items())
    for i, (k, v) in enumerate(items):
        last_child = (i == len(items) - 1)
        dump_structure(
            obj=v,
            name=str(k),
            prefix=new_prefix,
            is_last=last_child,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            file=file
        )