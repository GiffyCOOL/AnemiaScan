# AnemiaScan on the Raspberry Pi

The camera (USB) is plugged into the Mac and looks into the box. The browser on the Mac takes the photo and
sends it to the Raspberry Pi, which runs all four models, controls the LED and sends the result back to the page.

```
 USB camera ──> Mac (browser, http://localhost:5051) <── Ethernet cable / SSH link ──> Raspberry Pi (app.py) ── GPIO ──> LED
```
If the Pi Camera port works one day, plug the Pi Camera in instead: the app uses it automatically.

## A. On the Mac (already done once)

| Double-click | What it does |
|---|---|
| `Mac_Setup.command` | One time: Python packages for testing, Raspberry Pi Imager, SSH key. |
| `Test_on_Mac.command` | Runs the app on the Mac (Mac camera or "upload photo") at http://localhost:5050. |
| `Send_to_Pi.command` | Copies everything to the Pi, installs it there, and makes the app start whenever the Pi turns on. |
| `Open_AnemiaScan.command` | **Daily use**: connects to the Pi and opens the app at http://localhost:5051 (so the browser may use the Mac's camera). Keep its window open. |
| `Pi_Tools.command` | Open the app, live log, restart, LED blink test, camera test, Pi terminal, safe shutdown. |
| `Open AnemiaScan.webloc` | Opens http://localhost:5051 (only while `Open_AnemiaScan.command` is running) |

If macOS says a `.command` file "can't be opened": right-click it → **Open** → **Open** (first time only).

## B. Hardware day (once)

**1. Write the SD card** (open **Raspberry Pi Imager** from Applications, SD card in the Mac):
- Device: your Pi model (printed on the board, e.g. "Raspberry Pi 4 Model B"). Not sure? Pick "No filtering".
- OS: **Raspberry Pi OS (64-bit)**. Storage: the SD card.
- **Edit settings**:
  - Hostname `anemiascan`
  - Username `pi` and a password you choose
  - Wi-Fi: leave empty (the Ethernet cable is used)
  - Services: **Enable SSH** → **Allow public-key authentication only** (your Mac's key is filled in)
- Write. Takes about 5–10 minutes.

**2. Plug in** (Pi switched off):
- USB camera into the **Mac**, fixed in the box looking at the hand.
- LED wires on the header (see C).
- SD card into the Pi.
- Ethernet cable from the Pi to the Mac's USB-Ethernet adapter.

**3. Give the Pi internet through the Mac** (the first install downloads packages):
System Settings → General → **Sharing** → **Internet Sharing** (i) →
"Share your connection from" **Wi-Fi**, "To devices using" your **USB LAN / Ethernet adapter** → Done → switch Internet Sharing **on**.

**4. Power on the Pi**, wait 2–3 minutes, then double-click **`Send_to_Pi.command`**.
After about 5 minutes the browser opens AnemiaScan. From then on the app starts by itself whenever the Pi is powered;
open it with **`Open_AnemiaScan.command`**. The first time, allow the browser to use the camera, and pick the USB camera in the list under the live view.

## C. The LED

Look where the LED's two wires go on the 40-pin header:
- **GPIO pin + GND** (for example physical pin 11 = GPIO 17): the app switches it on. Check with `Pi_Tools.command` → 4.
  If your pin is not GPIO 17, change `LED_PIN` at the top of `app.py` and run `Send_to_Pi.command` again.
- **3.3 V / 5 V pin + GND** (physical pins 1, 2, 4, 17): it is always on. Set `LED_PIN = None` (optional).

An LED on a GPIO pin needs a resistor (about 220–330 Ω) in series.

## D. Using it

1. Fingers flat in the box → **Capture NAIL**. Check the green nail region.
2. Open palm facing the camera → **Capture PALM**.
3. Gender and age (type it in years) → **Analyze**: fused Hb, verdict, severity, palm and nail values.
4. **New person** clears everything.

After warm-up the page tries to **lock** the camera's exposure and white balance (the tag under the live view says "colours locked" or "auto colours"; Chrome supports this on more cameras than Safari). Keep the box closed and the LED the only light, so every photo is taken under the same light.
Turn off with `Pi_Tools.command` → 7 before unplugging.

## E. Files

| File | What it does |
|---|---|
| `app.py` | The web app. Settings at the top (`LED_PIN`, camera size). |
| `anemiascan_core.py` | v3 pipeline: segmentation → color features → Hb (Random Forest) → fusion palm 0.503 / nail 0.497 → WHO verdict (F < 12, M < 13 g/dL). |
| `templates/index.html` | The page. |
| `models/` | 4 ONNX models + `config.json`, made by `Export_Models_for_Pi.ipynb` from the final models. |
| `setup_pi.sh`, `anemiascan.service`, `led_test.py` | Used on the Pi by `Send_to_Pi.command` / `Pi_Tools.command`. |

## F. Problems

| Problem | Fix |
|---|---|
| "Pi not found" | Internet Sharing on? Cable in? Wait 3 min after power-on. Try again. |
| "camera blocked" on the page | Open the app with `Open_AnemiaScan.command` (localhost), not anemiascan.local. Allow camera access when asked. |
| Wrong camera shown | Pick the USB camera in the list under the live view (remembered next time). |
| "no nail/palm found" | Hand closer, box closed, retake. |
| Palm and nail differ by > 1.5 g/dL | Retake both photos. |
| Test on Mac: no camera | System Settings → Privacy & Security → Camera → allow Terminal. Or use "upload photo". |

*Research prototype, not a medical diagnosis.*
