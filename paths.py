"""Project-root-relative file locations for persisted local state.

Kept as a single flat module (not inside a package) so PROJECT_ROOT always
resolves to the actual project root regardless of which subpackage imports
it - gestures/, amp/, effects/ etc. can all just import these constants
instead of each computing its own __file__-relative path (which would
otherwise point at their own subdirectory, not the project root).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"
MOUTH_CALIBRATION_FILE = PROJECT_ROOT / "mouth_calibration.json"
GAIN_FILE = PROJECT_ROOT / "loop_gain.json"
MAC_GAIN_FILE = PROJECT_ROOT / "mac_gain.json"
MONITOR_GAIN_FILE = PROJECT_ROOT / "monitor_gain.json"
