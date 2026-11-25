#!/bin/bash
# Runs AnemiaScan on this Mac (Mac camera or uploaded photos). Close this window or press Ctrl+C to stop.
cd "$(dirname "$0")"
[ -x .venv-mac/bin/python ] || { echo "Run Mac_Setup.command first."; read -p "Press Enter to close"; exit 1; }
PORT=5050   # 5000 is used by macOS AirPlay; the browser uses the Mac camera
(sleep 6; open "http://localhost:$PORT") &
.venv-mac/bin/python app.py --port $PORT --no-camera "$@"
