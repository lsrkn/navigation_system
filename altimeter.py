import time
from collections import deque

class Altimeter:
    """Оценка высоты над землей. Режимы: 'sim' | 'bmp' | 'laser'."""

    def __init__(self, mode="sim", filter_window=5):
        self.mode = mode
        self.buffer = deque(maxlen=filter_window)
        self.sim_altitude = 10.0  # высота по умолчанию для отладки

        if mode == "bmp":
            import smbus2                      # pip install smbus2
            self.bus = smbus2.SMBus(1)
            self._init_bmp()                   # инициализация BMP3xx по даташиту
        elif mode == "laser":
            import serial                      # pip install pyserial
            self.ser = serial.Serial("/dev/ttyAMA0", 115200, timeout=1)

    def read_raw(self):
        if self.mode == "sim":
            return self.sim_altitude
        if self.mode == "bmp":
            return self._read_bmp()            # барометрическая формула, давление -> метры
        return self._read_laser()              # парсинг кадра TF-Luna

    def get_altitude(self):
        """Медианный фильтр по окну — глушит шум барометра и выбросы."""
        self.buffer.append(self.read_raw())
        return sorted(self.buffer)[len(self.buffer) // 2]

    def set_sim_altitude(self, h):             # только для симуляции/тестов
        self.sim_altitude = h
