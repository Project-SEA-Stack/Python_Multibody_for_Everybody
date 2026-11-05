# source/multibody/_cli_support.py
"""
@author: Sahand Sabet

Small, shared helpers for the repo's main_*.py entry scripts -- not part of
the public `multibody` API (no re-export in multibody/__init__.py).
"""
import sys

_HELP_FOOTER = (
    "\n" + "-" * 72 + "\n"
    "Hit a problem? Check documentation/build/html/index.html or open an issue:\n"
    "https://github.com/Project-SEA-Stack/Python_multibody_dyamics/issues\n"
    "(please include the traceback above)\n"
    + "-" * 72
)


def install_error_footer():
    """
    Install a sys.excepthook that prints the normal traceback UNCHANGED,
    then appends a short pointer to the docs/issue tracker -- never replaces
    or shortens the traceback, and never fires for a plain KeyboardInterrupt
    (Ctrl+C is a deliberate user action, not a bug).
    """
    default_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        default_hook(exc_type, exc_value, exc_tb)
        if issubclass(exc_type, KeyboardInterrupt):
            return
        print(_HELP_FOOTER)

    sys.excepthook = _hook
