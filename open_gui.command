#!/bin/bash
cd "$(dirname "$0")"
pkill -f "[P]ython.*gui.py" 2>/dev/null || true
pkill -f "[p]ython3 gui.py" 2>/dev/null || true
export AERO_GUI_LAUNCHED=1
exec python3 gui.py
