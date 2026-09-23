#!/usr/bin/env python3
"""
Browser front end for the Aero CFD solver.

    python3 webui.py                 # http://localhost:8017
    python3 webui.py --port 9000
    python3 webui.py --host 0.0.0.0  # reachable from the local network

Serves a local web UI for setting up a case, running it with live progress,
and reading the coefficients and flow field back.  Needs nothing beyond the
package's own dependencies — no Flask, no PySide6, no display server — so it
works over SSH and inside containers where the Qt app cannot run.
"""

import argparse
import sys
import webbrowser

from aero.web.server import serve


def main() -> int:
    p = argparse.ArgumentParser(description="Aero CFD — local web UI")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (0.0.0.0 exposes it to your network)")
    p.add_argument("--port", type=int, default=8017, help="port to listen on")
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = p.parse_args()

    if not args.no_open and args.host in ("127.0.0.1", "localhost"):
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
