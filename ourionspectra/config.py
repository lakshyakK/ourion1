"""
Central place for colors, layout constants, and the atmospheric feature
table. Import from here instead of hardcoding hex values / magic numbers
around the codebase.
"""

import os
import sys

if getattr(sys, "frozen", False):
    # PyInstaller extracts bundled 'datas' (like assets/) into sys._MEIPASS,
    # which for onedir builds is the '_internal' folder next to the exe -
    # NOT the same folder as the exe itself. Use _MEIPASS when present.
    APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
else:
    # .../ourionspectra/ourionspectra/config.py -> project root is two levels up
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- COLORS
NAVY = "#0f2a52"
NAVY_DEEP = "#0a1d3a"
BLUE_ACCENT = "#3b6fb0"
BG = "#ffffff"
CARD_BG = "#ffffff"
TEXT_SUB = "#66727e"
GREEN = "#1f8a5b"
AMBER = "#b8791a"
GRAY_LINE = "#9aa5b1"
BORDER = "#e6e9ee"
STATUS_GREEN = "#1f8a5b"
STATUS_AMBER = "#c98a1a"
STATUS_GRAY = "#8b95a1"

# ---------------------------------------------------------------- LOGO
# Preferred: put your logo at <project root>/assets/logo.png (portable).
# Fallback: your original Desktop file, if the assets copy isn't present.
_ASSETS_LOGO = os.path.join(APP_DIR, "assets", "logo.png")
_ASSETS_WEBLOGO = os.path.join(APP_DIR, "assets", "weblogo-removebg-preview.png")

if os.path.exists(_ASSETS_LOGO):
    LOGO_PATH = _ASSETS_LOGO
elif os.path.exists(_ASSETS_WEBLOGO):
    LOGO_PATH = _ASSETS_WEBLOGO
else:
    LOGO_PATH = _ASSETS_LOGO  # fallback to text logo if neither exists


RANDOM_SEED = 2026
# ---------------------------------------------------------------- CSV IMPORT
# Wavelength unit -> multiplier to convert into nanometers.
WAVELENGTH_UNIT_TO_NM = {
    "nm (nanometers)": 1.0,
    "µm (microns)": 1000.0,
    "Å (angstroms)": 0.1,
}
