"""
OURIONSPECTRA — Exoplanet Atmospheric Spectrum Recovery
Entry point. Run with:  python main.py

Requires:  pip install matplotlib pillow
(tkinter ships with standard Python on Windows/Mac; on Linux install
 python3-tk via your package manager if it's missing)
"""

import sys


def _check_dependencies():
    missing = []
    try:
        import tkinter  # noqa: F401
    except ImportError:
        missing.append("tkinter (on Linux: sudo apt install python3-tk)")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib (pip install matplotlib)")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow (pip install pillow)")
    if missing:
        print("Missing required packages:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall everything with:  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    _check_dependencies()
    from ourionspectra.app import OurionSpectraApp

    app = OurionSpectraApp()
    app.mainloop()

