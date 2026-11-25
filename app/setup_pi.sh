#!/bin/bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS). Run inside this folder:  bash setup_pi.sh
set -e
cd "$(dirname "$0")"
echo "== 1/4 System packages (camera, OpenCV, Flask, GPIO) =="
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-flask python3-gpiozero python3-numpy python3-venv
echo "== 2/4 Python environment =="
python3 -m venv --system-site-packages .venv          # can see the apt packages above
.venv/bin/pip install --upgrade pip
.venv/bin/pip install onnxruntime                     # runs the 4 models (keeps the apt numpy)
echo "== 3/4 Checks =="
.venv/bin/python -c "import onnxruntime, cv2, flask, picamera2; print('libraries OK, onnxruntime', onnxruntime.__version__)"
for f in nail_seg.onnx palm_seg.onnx nail_hb.onnx palm_hb.onnx config.json; do
  [ -f "models/$f" ] && echo "  models/$f OK" || echo "  models/$f MISSING  -> copy the models folder here"
done
rpicam-hello --list-cameras || echo "  (no camera listed: check the ribbon cable)"
echo "== 4/4 Done =="
echo "Start the app:   .venv/bin/python app.py"
echo "Then on the Mac open:   http://$(hostname).local:5000"
