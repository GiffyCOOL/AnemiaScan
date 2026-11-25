"""AnemiaScan GUI server for the Raspberry Pi.

Run on the Pi:   python app.py
Open on the Mac: http://<pi-name>.local:5000   (for example http://anemiascan.local:5000)

The Pi takes the photos (Pi Camera + LED), runs all the models and sends the result to the browser.
Without a camera (for example on a laptop) the page still works with "Upload photo".
"""
import argparse, atexit, base64, os, threading, time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response

from anemiascan_core import AnemiaScan, overlay, who_status

# Mean Hb change (g/dL) vs ages 18-25, by sex: NHANES 2015-2018 (CDC), n=10,587 adults, survey-weighted, smoothed.
_AGE_PTS = [22, 28, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82, 85]
_AGE_OFF = {"male":   [0, -0.10, -0.15, -0.15, -0.21, -0.22, -0.27, -0.38, -0.48, -0.63, -0.72, -1.11, -1.27, -1.70],
            "female": [0, -0.05, -0.06, -0.11, -0.10, 0.00, 0.22, 0.35, 0.37, 0.37, 0.32, 0.14, 0.00, -0.22]}
MODEL_AGE = 25.0                                    # top of the training age range


def age_offset(age, sex):
    return float(np.interp(age, _AGE_PTS, _AGE_OFF[sex]))

# ── settings you may need to change ───────────────────────────────────────────
LED_PIN = 17                 # BCM GPIO number the LED is wired to; None if the LED is wired to a power pin (always on)
CAPTURE_SIZE = (1640, 1232)  # camera resolution used for preview and photos
LOCK_CAMERA = True           # after warm-up, freeze exposure and white balance so every photo has the same colors
WARMUP_SECONDS = 2.0
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")   # every captured photo is saved here


class Light:
    """The LED in the box. On while the app runs (the preview and the photos need it), off when the app stops."""
    def __init__(self, pin):
        self.led = None
        if pin is None:
            print("LED: not controlled (LED_PIN = None)"); return
        try:
            from gpiozero import LED
            self.led = LED(pin)
            print(f"LED: GPIO {pin}")
        except Exception as e:
            print(f"LED: could not use GPIO {pin} ({e}); continuing without it")
    def on(self):
        if self.led: self.led.on()
    def off(self):
        if self.led: self.led.off()


class Camera:
    """Pi Camera through picamera2; falls back to a USB/laptop webcam; or no camera (upload only)."""
    def __init__(self):
        self.kind, self.cam, self.lock = None, None, threading.Lock()
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"}))   # RGB888 = BGR pixel order, as OpenCV uses
            cam.start()
            self.cam, self.kind = cam, "picamera2"
        except Exception as e:
            print(f"Pi Camera not available ({e})")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
                self.cam, self.kind = cap, "webcam"
        print(f"Camera: {self.kind or 'none (use Upload photo)'}")

    locked = False

    def unlock(self):
        """Back to automatic exposure and white balance (used between people)."""
        if self.kind == "picamera2" and self.locked:
            self.cam.set_controls({"AeEnable": True, "AwbEnable": True}); self.locked = False
            print("Camera settings unlocked (auto)")

    def lock_settings(self):
        """Fix exposure and white balance at the values the camera chose under the LED."""
        if self.kind != "picamera2" or not LOCK_CAMERA or self.locked:
            return
        self.locked = True
        md = self.cam.capture_metadata()
        controls = {"AeEnable": False, "AwbEnable": False}
        for k in ("ExposureTime", "AnalogueGain", "ColourGains"):
            if k in md: controls[k] = md[k]
        self.cam.set_controls(controls)
        print("Camera settings locked:", {k: v for k, v in controls.items() if k not in ("AeEnable", "AwbEnable")})

    def frame(self):
        with self.lock:
            if self.kind == "picamera2":
                return self.cam.capture_array("main")
            if self.kind == "webcam":
                ok, img = self.cam.read()
                return img if ok else None
        return None

    def close(self):
        if self.kind == "picamera2": self.cam.stop()
        elif self.kind == "webcam": self.cam.release()


def jpg_b64(img, max_side=900):
    h, w = img.shape[:2]; k = min(1.0, max_side / max(h, w))
    if k < 1: img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


app = Flask(__name__)
STATE = {}          # region -> {"img", "mask", "roi", "box", "found"}
MODEL = CAMERA = LIGHT = None


@app.route("/")
def index():
    return render_template("index.html", camera=CAMERA.kind if CAMERA else None)


@app.route("/preview.jpg")
def preview():
    img = CAMERA.frame() if CAMERA else None
    if img is None:
        return Response(status=204)
    h, w = img.shape[:2]; k = 640 / max(h, w)
    small = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return Response(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/capture/<region>", methods=["POST"])
def capture(region):
    if region not in ("nail", "palm"):
        return jsonify(error="region must be nail or palm"), 400
    if "file" in request.files and request.files["file"].filename:                 # uploaded photo
        img = cv2.imdecode(np.frombuffer(request.files["file"].read(), np.uint8), cv2.IMREAD_COLOR)
    else:                                                                            # photo from the camera
        if CAMERA and CAMERA.kind == "picamera2" and not CAMERA.locked:
            CAMERA.lock_settings(); time.sleep(0.3)     # freeze colors now (box closed, LED on) for both photos of this person
        img = CAMERA.frame() if CAMERA else None
    if img is None:
        return jsonify(error="No photo: the camera is not available, use Upload photo"), 400
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{region}.jpg"), img)
    except Exception as e:
        print("could not save capture:", e)
    t = time.time()
    found = MODEL.find_region(region, img)
    STATE[region] = dict(found, img=img)
    shown = overlay(img, found["mask"], found["box"]) if found["found"] else img
    return jsonify(found=found["found"], image=jpg_b64(shown), seconds=round(time.time() - t, 2))


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    sex = data.get("sex")
    if sex not in ("female", "male"):
        return jsonify(error="Choose a gender"), 400
    try:
        age = float(data.get("age"))
    except (TypeError, ValueError):
        return jsonify(error="Type the age in years"), 400
    if not 5 <= age <= 100:
        return jsonify(error="Age must be between 5 and 100 years"), 400
    if not STATE:
        return jsonify(error="Capture the nail and palm photos first"), 400
    t = time.time()
    res = MODEL.screen(STATE, sex, MODEL_AGE if age > MODEL_AGE else age)
    if res is None:
        return jsonify(error="Neither a nail nor a palm was detected. Retake the photos."), 400
    if age > MODEL_AGE:
        d = age_offset(age, sex) - age_offset(MODEL_AGE, sex)
        for k in ("hb", "hb_palm", "hb_nail"):
            if res.get(k) is not None: res[k] += d
        res["anemic"], res["severity"] = who_status(res["hb"], sex)
    elif age < 18:
        res["warnings"].append("The models were trained on young adults (mostly 18-25), so this result is less reliable.")
    res.update(sex=sex, age=f"{age:g}", seconds=round(time.time() - t, 2))
    print(f"[result] {sex}, age {age:g}: palm {res['hb_palm']}, nail {res['hb_nail']} -> {res['hb']:.2f} g/dL, {res['severity']}")
    return jsonify(res)


@app.route("/last/<region>.jpg")
def last_photo(region):
    """The last raw photo of a region (for checking camera and lighting)."""
    if region not in STATE:
        return Response(status=404)
    return Response(cv2.imencode(".jpg", STATE[region]["img"], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/reset", methods=["POST"])
def reset():
    STATE.clear()
    if CAMERA: CAMERA.unlock()
    return jsonify(ok=True)


def main():
    global MODEL, CAMERA, LIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-camera", action="store_true", help="upload photos only")
    args = ap.parse_args()
    print("Loading models ...")
    MODEL = AnemiaScan()
    LIGHT = Light(None if args.no_camera else LED_PIN); LIGHT.on()
    CAMERA = None if args.no_camera else Camera()
    if CAMERA and CAMERA.kind:
        time.sleep(WARMUP_SECONDS)     # colors are locked at the first capture of each person, not here
    atexit.register(lambda: (LIGHT.off(), CAMERA and CAMERA.close()))
    print(f"\nAnemiaScan is running. On the Mac open:  http://<this-pi-name>.local:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
