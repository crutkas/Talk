"""Auto-install Python packages at runtime.

Handles pip installing dependencies when a user selects a model
that requires packages not yet installed.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is importable."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_packages(
    packages: list[str],
    progress_callback: Any | None = None,
) -> bool:
    """Install Python packages via pip.

    Args:
        packages: List of pip package specifiers (e.g. ["faster-whisper>=1.0"]).
        progress_callback: Optional callable(status: str) for UI updates.

    Returns:
        True if all packages installed successfully.
    """
    if not packages:
        return True

    pkg_list = ", ".join(packages)
    logger.info("Installing packages: %s", pkg_list)
    if progress_callback:
        progress_callback(f"📦 Installing {pkg_list}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *packages],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error("pip install failed: %s", result.stderr)
            if progress_callback:
                progress_callback(f"❌ Failed to install {pkg_list}")
            return False

        logger.info("Successfully installed: %s", pkg_list)
        if progress_callback:
            progress_callback(f"✅ Installed {pkg_list}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("pip install timed out for: %s", pkg_list)
        if progress_callback:
            progress_callback(f"❌ Install timed out for {pkg_list}")
        return False
    except Exception as e:
        logger.error("pip install error: %s", e)
        if progress_callback:
            progress_callback(f"❌ Install error: {e}")
        return False


def ensure_packages(
    packages: dict[str, str],
    progress_callback: Any | None = None,
) -> bool:
    """Check and install packages if missing.

    Args:
        packages: Dict of {import_name: pip_specifier}.
            e.g. {"faster_whisper": "faster-whisper>=1.0"}
        progress_callback: Optional callable for UI updates.

    Returns:
        True if all packages are available (already installed or newly installed).
    """
    missing = []
    for import_name, pip_spec in packages.items():
        if not is_package_installed(import_name):
            missing.append(pip_spec)

    if not missing:
        return True

    return install_packages(missing, progress_callback)
