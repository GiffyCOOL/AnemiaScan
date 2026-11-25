#!/bin/bash
# Copies this folder to the Raspberry Pi, installs everything there and starts AnemiaScan.
# Use it once the Pi is plugged in and switched on (first boot takes about 2-3 minutes).
cd "$(dirname "$0")"
PI="${PI:-pi@anemiascan.local}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
echo "Looking for the Pi ($PI) ..."
HOST="${PI#*@}"
for i in $(seq 1 36); do nc -z -G 3 "$HOST" 22 2>/dev/null && break
  [ $i = 36 ] && { echo "!! Pi not found. Check the cable, Internet Sharing and that it has booted (wait 3 min after power-on)."; read -p "Press Enter to close"; exit 1; }
  sleep 5; done
if ! $SSH -o BatchMode=yes "$PI" true 2>/dev/null; then
  echo "Adding this Mac's key to the Pi (type the Pi password once) ..."
  ssh-copy-id -o StrictHostKeyChecking=accept-new "$PI" || { read -p "Login failed. Press Enter"; exit 1; }
fi
echo "Pi found. Copying files ..."
COPYFILE_DISABLE=1 tar --exclude .venv-mac --exclude .venv --exclude __pycache__ --exclude '*.command' --exclude '*.webloc' -czf - . \
  | $SSH "$PI" 'mkdir -p ~/AnemiaScan_Pi_GUI && tar -xzf - -C ~/AnemiaScan_Pi_GUI' || { read -p "Copy failed. Press Enter"; exit 1; }
echo "Installing on the Pi (about 5 minutes the first time) ..."
$SSH -t "$PI" 'cd ~/AnemiaScan_Pi_GUI && bash setup_pi.sh && sed -i "s#/home/pi#$HOME#g; s#User=pi#User=$USER#" anemiascan.service && sudo cp anemiascan.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable anemiascan && sudo systemctl restart anemiascan && sleep 8 && systemctl --no-pager --lines=15 status anemiascan'
echo
echo "AnemiaScan now starts by itself every time the Pi is switched on."
echo "Opening it now (next time: double-click Open_AnemiaScan.command) ..."
bash "$(dirname "$0")/Open_AnemiaScan.command"
read -p "Press Enter to close"
