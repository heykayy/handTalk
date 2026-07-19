"""
app_paths.py – single source of truth for "where is the app installed"
=========================================================================
Both `utils.py` (word model) and `sentence/sentence_model.py` (sentence
model) need to know the folder that contains `word/` and `sentence/` so
they can find the .keras models and label maps.

When running from source that folder is simply the repo root. When running
from a PyInstaller-built executable it is NOT the same as the script's
`__file__` location any more:

  * --onefile builds unpack everything into a temp dir exposed as
    sys._MEIPASS at runtime.
  * --onedir builds keep everything next to the .exe, so the app root is
    the folder holding sys.executable.

`app_root()` returns the right one automatically, so the rest of the
codebase can keep using plain relative paths ("word/isl_final_model.keras",
etc.) whether it's being run with `python predict.py` or as a packaged
desktop app downloaded from GitHub Releases.
"""

import os
import sys


def app_root() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    # Source checkout: this file lives at the repo root, next to word/ and
    # sentence/, so its own directory *is* the app root.
    return os.path.dirname(os.path.abspath(__file__))
