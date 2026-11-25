#!/bin/bash
# Small helper menu for the Pi in the box.
cd "$(dirname "$0")"
PI="${PI:-pi@anemiascan.local}"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
echo "1) Open AnemiaScan in the browser"
echo "2) Show the app's live log (Ctrl+C to leave)"
echo "3) Restart the app"
echo "4) Blink the LED on GPIO 17 (wiring test)"
echo "5) Test the camera"
echo "6) Open a terminal on the Pi"
echo "7) Shut the Pi down safely"
read -p "Choose 1-7: " n
case $n in
  1) bash "$(dirname "$0")/Open_AnemiaScan.command" ;;
  2) $SSH -t "$PI" 'journalctl -fu anemiascan' ;;
  3) $SSH "$PI" 'sudo systemctl restart anemiascan' && echo restarted ;;
  4) $SSH -t "$PI" 'cd ~/AnemiaScan_Pi_GUI && sudo systemctl stop anemiascan; .venv/bin/python led_test.py 17; sudo systemctl start anemiascan' ;;
  5) $SSH -t "$PI" 'sudo systemctl stop anemiascan; rpicam-hello --list-cameras; rpicam-still -n -t 1500 -o ~/camtest.jpg && echo "photo OK"; sudo systemctl start anemiascan' && scp "$PI":~/camtest.jpg /tmp/camtest.jpg && open /tmp/camtest.jpg ;;
  6) $SSH "$PI" ;;
  7) $SSH "$PI" 'sudo shutdown now'; echo "Wait for the green LED to stop, then unplug." ;;
esac
read -p "Press Enter to close"
