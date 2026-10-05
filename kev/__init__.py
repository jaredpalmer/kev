# PyPI ships a different, unrelated package under the name "kev".
# Detect that at import time so users get a clear error instead of
# mysterious AttributeErrors later.
import warnings as _w

try:
    from importlib.metadata import metadata as _meta

    _m = _meta("kev")
    _urls = str(_m.get("Home-page") or "") + str(_m.get("Project-URL") or "")
    if "jaredpalmer" not in _urls:
        _w.warn(
            "You have the wrong 'kev' package installed. "
            "PyPI's 'kev' is unrelated to jaredpalmer/kev. "
            "Install from source instead:\n"
            "  pip install git+https://github.com/jaredpalmer/kev",
            ImportWarning,
            stacklevel=2,
        )
    del _m, _urls, _meta
except Exception:
    pass

del _w
