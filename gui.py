#!/usr/bin/env python3
"""Desktop GUI launcher for Aero CFD."""

import os
import shlex
import subprocess
import sys

_repo_root = os.path.dirname(os.path.abspath(__file__))
_mpl_cache = os.path.join(_repo_root, ".cache", "matplotlib")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


def _running_inside_cursor() -> bool:
    return os.environ.get("TERM_PROGRAM") in {"vscode", "cursor"}


def _relaunch_in_terminal_app() -> int:
    """macOS GUI apps need a real window server session; Cursor's shell often can't show Qt windows."""
    shell_cmd = (
        f"cd {shlex.quote(_repo_root)} && "
        f"export AERO_GUI_LAUNCHED=1 && "
        f"exec {shlex.quote(sys.executable)} gui.py"
    )
    # AppleScript strings must use double quotes (not Python's single-quoted repr).
    escaped = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        'tell application "Terminal" to activate\n'
        f'tell application "Terminal" to do script "{escaped}"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Could not open Terminal.app automatically ({exc}).", flush=True)
        print(f"Double-click: {_repo_root}/open_gui.command", flush=True)
        return 1
    print("Opened Aero CFD Studio in Terminal.app.", flush=True)
    print("(Use that Terminal window for the GUI — this Cursor tab is done.)", flush=True)
    return 0


def main() -> int:
    if (
        sys.platform == "darwin"
        and _running_inside_cursor()
        and os.environ.get("AERO_GUI_LAUNCHED") != "1"
        and "--here" not in sys.argv
    ):
        return _relaunch_in_terminal_app()

    if "--here" in sys.argv and _running_inside_cursor():
        print(
            "  Running in Cursor terminal (--here). Interactive 3D uses matplotlib here.",
            flush=True,
        )

    print("Starting Aero CFD Studio...", flush=True)
    print("  Loading modules...", flush=True)
    from aero.gui.app import launch_gui

    print("  (Terminal will stay on Ready until you close the GUI window.)", flush=True)
    return launch_gui(_repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
