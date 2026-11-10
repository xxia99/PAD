"""Optional SwanLab compatibility layer.

The reviewer archive should not contact external logging services by default.
Set `PAD_ENABLE_SWANLAB=1` to use SwanLab if it is installed and desired.
"""

import os
import sys


class _NoOpSwanLab:
    def __init__(self, import_error):
        self.import_error = import_error
        self.config = {}

    def init(self, *args, **kwargs):
        print(f"[swanlab disabled] import failed: {self.import_error}", file=sys.stderr)
        return None

    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


if os.environ.get("PAD_ENABLE_SWANLAB", "0") == "1":
    try:
        import swanlab as swanlab  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host logger install
        swanlab = _NoOpSwanLab(exc)
else:
    swanlab = _NoOpSwanLab("external logging disabled; set PAD_ENABLE_SWANLAB=1 to enable")
