#!/bin/bash
# AnemiaScan: one-time setup of this Mac. Double-click it (or run: bash Mac_Setup.command).
cd "$(dirname "$0")"
echo "=== AnemiaScan Mac setup ==="
chmod +x ./*.command 2>/dev/null

# 1) Python environment for testing the app on the Mac
PY=""
for c in python3 /opt/anaconda3/bin/python3 "$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3" /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, venv; assert sys.version_info >= (3, 9)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "!! Python 3.9+ not found. Install it from python.org and run this again."; read -p "Press Enter to close"; exit 1; fi
echo "1/4 Python: $("$PY" --version) ($PY)"
[ -d .venv-mac ] || "$PY" -m venv .venv-mac
.venv-mac/bin/pip install -q --upgrade pip
.venv-mac/bin/pip install -q flask numpy opencv-python-headless onnxruntime && echo "    packages OK" || echo "!! package install failed (check the internet connection)"

# 2) Models (exported from Colab as AnemiaScan_Pi_models.zip into Downloads)
echo "2/4 Models:"
if [ ! -f models/nail_seg.onnx ] && [ -f "$HOME/Downloads/AnemiaScan_Pi_models.zip" ]; then
  unzip -oq "$HOME/Downloads/AnemiaScan_Pi_models.zip" -d models && rm -f models/PUT_MODELS_HERE.txt && echo "    unzipped from Downloads/AnemiaScan_Pi_models.zip"
fi
ok=1; for f in nail_seg.onnx palm_seg.onnx nail_hb.onnx palm_hb.onnx config.json; do
  if [ -f "models/$f" ]; then echo "    models/$f OK"; else echo "    models/$f MISSING"; ok=0; fi; done

# 3) Raspberry Pi Imager (writes the SD card)
echo "3/4 Raspberry Pi Imager:"
if [ -d "/Applications/Raspberry Pi Imager.app" ]; then echo "    already installed"
else
  curl -fL --progress-bar -o /tmp/rpi-imager.dmg https://downloads.raspberrypi.com/imager/imager_latest.dmg \
  && hdiutil attach -nobrowse -quiet -mountpoint /tmp/rpi-imager-vol /tmp/rpi-imager.dmg \
  && cp -R /tmp/rpi-imager-vol/*.app /Applications/ && echo "    installed in Applications" || echo "!! could not install Imager: download it from raspberrypi.com/software"
  hdiutil detach -quiet /tmp/rpi-imager-vol 2>/dev/null; rm -f /tmp/rpi-imager.dmg
fi

# 4) SSH key, so the Mac can talk to the Pi without typing the password every time
echo "4/4 SSH key:"
if ls "$HOME"/.ssh/id_ed25519.pub >/dev/null 2>&1 || ls "$HOME"/.ssh/id_rsa.pub >/dev/null 2>&1; then echo "    already have one"
else mkdir -p "$HOME/.ssh" && ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -C "anemiascan-mac" && echo "    created (Imager will offer it automatically)"; fi

echo
echo "Done. Next: double-click 'Test_on_Mac.command' to try the app with the Mac camera."
[ $ok = 1 ] || echo "(The app needs the 5 model files in the models folder first.)"
read -p "Press Enter to close"
