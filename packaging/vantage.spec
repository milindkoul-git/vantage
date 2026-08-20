# PyInstaller build specification.
#
# Produces a directory bundle containing vantage.exe and everything it needs to
# run on a machine with no Python. Built with:
#
#     pip install -e ".[dev,detect,package]"
#     pyinstaller packaging/vantage.spec --noconfirm
#
# Why a directory and not one file
# --------------------------------
# --onefile unpacks the entire bundle to a temporary directory on every launch.
# At roughly 450 MB that is several seconds of disk churn before the first frame
# and a copy left behind on an unclean exit. A directory bundle starts
# immediately and is what an installer would lay down anyway. The trade is that
# it is a folder rather than a single icon, which a zip solves.
#
# The parts PyInstaller cannot work out by itself
# -----------------------------------------------
# Three things here are not discoverable by static analysis, and each one fails
# at *runtime* rather than at build time, which is what makes them worth writing
# down:
#
# 1. **OpenVINO's plugin DLLs.** The runtime loads them by reading plugins.xml
#    at startup and dlopening whatever it names. Nothing imports them, so the
#    analyser never sees them, and the failure is "device GPU not found" on a
#    machine that plainly has one.
# 2. **The dashboard page.** It is data, not code.
# 3. **The adapter and backend modules.** They are resolved by name through the
#    registries rather than imported directly, so they must be named as hidden
#    imports or the catalog resolves to a module that is not in the bundle.
# 4. **The package metadata.** The version is read from it at import time, so a
#    bundle without it can build cleanly and still misreport its own build.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"

# --- data ------------------------------------------------------------------

datas = [
    (str(SRC / "vantage" / "dashboard" / "static"), "vantage/dashboard/static"),
    (str(ROOT / "configs" / "default.yaml"), "configs"),
]

# The package's own dist-info. ``vantage.__version__`` reads it through
# importlib.metadata rather than repeating the number in source, so without this
# the bundle answers ``vantage.exe --version`` with its unknown-version fallback
# - a shipped binary that cannot say which build it is.
datas += copy_metadata("vantage")

binaries = []
hiddenimports = [
    # Resolved by name from the adapter and backend registries.
    "vantage.perception.adapters.yolox",
    "vantage.perception.adapters.dfine",
    "vantage.perception.adapters.grounding_dino",
    "vantage.perception.backends.onnxruntime_backend",
    "vantage.perception.backends.openvino_backend",
    "vantage.pose.adapter",
    # Used through sqlite3 but sometimes missed on trimmed builds.
    "sqlite3",
]

# --- OpenVINO --------------------------------------------------------------
# The plugin DLLs and the manifest that names them. Without both, the runtime
# starts and then reports that no device exists.
try:
    import openvino

    openvino_root = Path(openvino.__file__).parent
    binaries += collect_dynamic_libs("openvino")
    for manifest in openvino_root.rglob("plugins.xml"):
        datas.append((str(manifest), str(manifest.parent.relative_to(openvino_root.parent))))
    hiddenimports += collect_submodules("openvino")
except ImportError:  # pragma: no cover - a CPU-only build is legitimate
    print("NOTE: openvino not installed; the bundle will be CPU-only", file=sys.stderr)

try:
    import onnxruntime  # noqa: F401

    binaries += collect_dynamic_libs("onnxruntime")
except ImportError:  # pragma: no cover
    print("NOTE: onnxruntime not installed", file=sys.stderr)

# --- what to leave out -----------------------------------------------------
# Development tooling and the plotting/GUI stacks numpy and OpenCV can drag in.
# Each of these is tens of megabytes for something the runtime never calls.
excludes = [
    "pytest",
    "mypy",
    "ruff",
    "matplotlib",
    "tkinter",
    "IPython",
    "notebook",
    "PIL",
    "scipy",
    "pandas",
]

a = Analysis(
    [str(ROOT / "packaging" / "entrypoint.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vantage",
    debug=False,
    strip=False,
    upx=False,  # UPX corrupts some OpenVINO DLLs and saves little on this bundle
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="vantage",
)
