"""Find out how the LED is wired.  Run:  .venv/bin/python led_test.py 17
It blinks the LED on that GPIO number 5 times. If it blinks, set LED_PIN in app.py to that number.
If the LED is on all the time, even with the Pi's apps stopped, it is wired to a power pin: set LED_PIN = None."""
import sys, time
from gpiozero import LED
pin = int(sys.argv[1]) if len(sys.argv) > 1 else 17
led = LED(pin)
print(f"Blinking GPIO {pin} ...")
for _ in range(5):
    led.on(); time.sleep(0.5); led.off(); time.sleep(0.5)
print("Done. Did it blink?")
