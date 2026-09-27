# path_buffer.py
"""LIFO-буфер маршрута: запись прямого пути и выдача точек
для обратного маршрута.

Буфер не зависит от способа локализации: ему подаются готовые
координаты (x, y) из системы позиционирования.
"""

import math
import pickle
import os


class PathBuffer:
    def __init__(self, min_distance=1.0, max_points=5000, reached_radius=0.8):
        """
        min_distance   — минимальная дистанция (м), после которой точка пишется в буфер
        max_points     — лимит буфера; при превышении путь прореживается вдвое
        reached_radius — радиус (м), в котором точка возврата считается пройденной
        """
        self.stack = []                  # [(x, y), ...] — от старта к текущей точке
        self.min_distance = min_distance
        self.max_points = max_points
        self.reached_radius = reached_radius

        self.return_mode = False         # True — идём назад

    # ---------------- ПРЯМОЙ ПУТЬ ----------------

    def record(self, pos):
        """Добавить точку. pos = (x, y). True — если точка записана."""
        if self.return_mode:
            return False
        if not self.stack:
            self._append(pos)
            return True
        if self._dist(pos, self.stack[-1]) >= self.min_distance:
            self._append(pos)
            return True
        return False

    def _append(self, pos):
        self.stack.append(tuple(pos))
        # защита от переполнения: прореживаем путь вдвое, сохраняя начало и конец
        if len(self.stack) > self.max_points:
            last = self.stack[-1]
            self.stack = self.stack[::2]
            if self.stack[-1] != last:
                self.stack.append(last)

    # ---------------- ОБРАТНЫЙ ПУТЬ ----------------

    def start_return(self, current_pos=None):
        """Переключиться в режим возврата. Упрощает путь.
        Возвращает количество точек возврата."""
        self.return_mode = True
        if not self.stack:
            return 0
        # если дрон уже у последней точки — не лететь к ней
        if current_pos is not None and \
           self._dist(current_pos, self.stack[-1]) < self.reached_radius:
            self.stack.pop()
        self.stack = self._simplify(self.stack)
        return len(self.stack)

    def next_waypoint(self):
        """Текущая цель возврата или None (буфер пуст = вернулись домой).
        Точка НЕ удаляется — удаление только после confirm_reached()."""
        if not self.return_mode or not self.stack:
            return None
        return self.stack[-1]

    def confirm_reached(self):
        """Удалить текущую цель из буфера. Вызывать, когда дрон
        надёжно находится в точке (проверено локализацией)."""
        if self.return_mode and self.stack:
            return self.stack.pop()
        return None

    # ---------------- СОСТОЯНИЕ ----------------

    @property
    def is_returning(self):
        return self.return_mode

    def __len__(self):
        return len(self.stack)

    # ---------------- СОХРАНЕНИЕ ----------------

    def save(self, filepath="path_stack.pkl"):
        """Атомарная запись на диск — файл не побьётся при сбое питания."""
        tmp = filepath + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.stack, f)
        os.replace(tmp, filepath)

    def load(self, filepath="path_stack.pkl"):
        if not os.path.exists(filepath):
            return False
        with open(filepath, "rb") as f:
            self.stack = pickle.load(f)
        return True

    # ---------------- ГЕОМЕТРИЯ ----------------

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _simplify(self, path, epsilon=None):
        """Итеративный Дуглас-Пекер: убирает точки, лежащие почти на прямой.
        Сокращает обратный маршрут без изменения формы более чем на epsilon."""
        if epsilon is None:
            epsilon = self.min_distance * 0.5
        n = len(path)
        if n < 3:
            return list(path)

        keep = [False] * n
        keep[0] = keep[-1] = True
        segments = [(0, n - 1)]

        while segments:
            s, e = segments.pop()
            if e <= s + 1:
                continue
            max_d, idx = 0.0, -1
            for i in range(s + 1, e):
                d = self._point_line_dist(path[i], path[s], path[e])
                if d > max_d:
                    max_d, idx = d, i
            if max_d > epsilon:
                keep[idx] = True
                segments.append((s, idx))
                segments.append((idx, e))

        return [p for p, k in zip(path, keep) if k]

    @staticmethod
    def _point_line_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
