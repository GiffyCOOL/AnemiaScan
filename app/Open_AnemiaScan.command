#!/bin/bash
# Opens AnemiaScan from the Pi through a secure local link, so the browser may use this Mac's camera.
# Keep this window open while you use the app; close it when you are done.
PI="${PI:-pi@anemiascan.local}"
PORT=5051
echo "Connecting to the Pi ($PI) ..."
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -f -N -L $PORT:localhost:5000 "$PI" || { echo "!! Pi not found. Is it on and connected? (see README)"; read -p "Press Enter to close"; exit 1; }
open "http://localhost:$PORT"
echo "AnemiaScan is open at http://localhost:$PORT  (close this window to disconnect)"
trap 'pkill -f "$PORT:localhost:5000"' EXIT
read -p "Press Enter to disconnect"
