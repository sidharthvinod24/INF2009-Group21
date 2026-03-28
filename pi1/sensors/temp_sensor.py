"""
sensors/temp_sensor.py
=======================
DHT22 / AM2302 temperature and humidity sensor reader for Pi 1.

Sensor is connected to GPIO 26.
Uses the Adafruit CircuitPython DHT library for reliable readings.

Readings are cached for TEMP_READ_INTERVAL seconds to avoid hammering
the sensor (DHT22 minimum read interval is ~2s, but we cache for longer
to reduce CPU overhead and sensor wear).

Graceful fallback:
    If adafruit_dht or board is not available (e.g. running on a dev machine),
    the module falls back to returning None values so the rest of the system
    continues running without crashing.

Install:
    pip install adafruit-circuitpython-dht
    sudo apt-get install libgpiod2
"""

import time
import threading

from config.settings import DHT_GPIO_PIN, TEMP_READ_INTERVAL


# ─── Optional hardware import ─────────────────────────────────────────────────

try:
    import adafruit_dht
    import board
    _DHT_AVAILABLE = True
except Exception as e:
    _DHT_AVAILABLE = False
    print(f'[Temp] adafruit_dht not available ({e}) — temperature sensor disabled.')


# ─── Sensor thread ────────────────────────────────────────────────────────────

class TempSensor:
    """
    Reads temperature and humidity from a DHT22 sensor on GPIO 26.
    Caches the last successful reading and updates it every TEMP_READ_INTERVAL seconds.

    Usage:
        sensor = TempSensor()
        sensor.start()
        temp, humidity = sensor.read()
        sensor.stop()
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._temperature = None   # degrees Celsius
        self._humidity    = None   # % relative humidity
        self._last_read   = 0.0
        self._error       = None
        self._stopped     = threading.Event()
        self._device      = None

        if _DHT_AVAILABLE:
            try:
                # Map GPIO pin number to board pin
                pin = getattr(board, f'D{DHT_GPIO_PIN}', None)
                if pin is None:
                    raise ValueError(f'board.D{DHT_GPIO_PIN} not found')
                self._device = adafruit_dht.DHT22(pin, use_pulseio=False)
                print(f'[Temp] DHT22 initialised on GPIO {DHT_GPIO_PIN}')
            except Exception as e:
                print(f'[Temp] DHT22 init failed: {e}')
                self._device = None

    def start(self):
        """Start background polling thread."""
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def stop(self):
        self._stopped.set()
        if self._device:
            try:
                self._device.exit()
            except Exception:
                pass

    # ── Public read API ───────────────────────────────────────────────────────

    def read(self):
        """
        Return (temperature_c, humidity_pct) from the last successful reading.
        Returns (None, None) if no reading has succeeded yet or sensor unavailable.
        """
        with self._lock:
            return self._temperature, self._humidity

    def read_formatted(self):
        """
        Return a human-readable string for Telegram or TTS.
        e.g. "Temperature: 28.5°C  Humidity: 65.2%"
        """
        temp, hum = self.read()
        if temp is None or hum is None:
            return 'Temperature sensor unavailable or no reading yet.'
        temp_f = (temp * 9 / 5) + 32
        feels  = self._feels_like(temp, hum)
        return (
            f'🌡 Temperature : {temp:.1f}°C  ({temp_f:.1f}°F)\n'
            f'💧 Humidity    : {hum:.1f}%\n'
            f'🌤 Feels like  : {feels}'
        )

    def last_read_age(self):
        """Seconds since the last successful sensor read."""
        return (time.time() - self._last_read) if self._last_read > 0 else float('inf')

    # ── Background polling ────────────────────────────────────────────────────

    def _poll_loop(self):
        if not _DHT_AVAILABLE or self._device is None:
            return
        while not self._stopped.is_set():
            self._read_once()
            self._stopped.wait(timeout=TEMP_READ_INTERVAL)

    def _read_once(self):
        if self._device is None:
            return
        try:
            temp = self._device.temperature
            hum  = self._device.humidity
            if temp is not None and hum is not None:
                with self._lock:
                    self._temperature = round(float(temp), 1)
                    self._humidity    = round(float(hum), 1)
                    self._error       = None
                self._last_read = time.time()
        except RuntimeError as e:
            # DHT22 commonly throws RuntimeError on bad reads — just retry
            with self._lock:
                self._error = str(e)
        except Exception as e:
            with self._lock:
                self._error = str(e)
            print(f'[Temp] Unexpected error: {e}')

    # ── Comfort helper ────────────────────────────────────────────────────────

    @staticmethod
    def _feels_like(temp_c, humidity):
        """Simple comfort description based on temperature and humidity."""
        if temp_c < 16:
            return 'Cold 🥶'
        elif temp_c < 20:
            return 'Cool'
        elif temp_c < 24:
            return 'Comfortable ✅'
        elif temp_c < 28:
            if humidity > 70:
                return 'Warm and humid'
            return 'Warm'
        elif temp_c < 32:
            if humidity > 70:
                return 'Hot and humid 🥵'
            return 'Hot'
        else:
            return 'Very hot ⚠️'
