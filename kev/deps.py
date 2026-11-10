"""Installed dependency checks, so a version mismatch names the requirement instead of a missing symbol.

kev.model imports `LinearAttentionCacheLayerMixin` from `transformers.cache_utils`; releases before 5.17 do not have it,
and pyproject allows `transformers>=5.17,<6`. A bundle with an older transformers used to fail on that import with
`ImportError: cannot import name 'LinearAttentionCacheLayerMixin'`, which does not say what to install (issue #280).
The check reads the installed distribution's metadata and never imports transformers itself.
"""

from importlib.metadata import PackageNotFoundError, version

TRANSFORMERS_REQUIREMENT = "transformers>=5.17,<6"  # pyproject.toml [project] dependencies, kept in sync by tests/test_deps.py
TRANSFORMERS_MIN = (5, 17)
TRANSFORMERS_MAX = (6,)


def _release(text):
    """The numeric release prefix of a version string: "5.17.0", "5.17.0+cpu" -> (5, 17, 0). A pre-release or local
    suffix is ignored (5.17.0.dev0 -> (5, 17, 0)); that is enough to tell 5.2 from 5.17, which is the mismatch here."""
    release = []
    for part in text.split("+")[0].split("."):
        if not part.isdigit():
            break
        release.append(int(part))
    return tuple(release)


def transformers_problem(found=None):
    """Why the installed transformers cannot run kev, or None. `found` is a version to check (default: the installed
    distribution's, from importlib.metadata). A missing distribution is not a version problem: the import that needs it
    raises its own ModuleNotFoundError, and nothing here hides that."""
    if found is None:
        try:
            found = version("transformers")
        except PackageNotFoundError:
            return None
    release = _release(found)
    if release and TRANSFORMERS_MIN <= release < TRANSFORMERS_MAX:
        return None
    return (
        f"Kev requires {TRANSFORMERS_REQUIREMENT}; found {found}. "
        f"Install with `pip install '{TRANSFORMERS_REQUIREMENT}'`."
    )


def require_transformers(found=None):
    """Raise ImportError naming pyproject's transformers range when the installed version is outside it."""
    problem = transformers_problem(found)
    if problem is not None:
        raise ImportError(problem)
