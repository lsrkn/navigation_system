# testing/test_path_buffer.py
"""Офлайн-тест буфера маршрута: запись с порогом, порядок возврата,
упрощение пути, переполнение, сохранение/загрузка."""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from path_buffer import PathBuffer


def test_record_threshold():
    pb = PathBuffer(min_distance=1.0)
    pb.record((0, 0))
    assert pb.record((0.5, 0)) is False      # слишком близко — не пишем
    assert pb.record((2.0, 0)) is True       # достаточно далеко — пишем
    assert len(pb) == 2

def test_return_order():
    pb = PathBuffer(min_distance=1.0)
    for i in range(10):
        pb.record((i * 2.0, 0))              # путь из 10 точек вдоль X
    pb.start_return(current_pos=(18.0, 0))   # последняя точка уже пройдена
    targets = []
    while True:
        t = pb.next_waypoint()
        if t is None:
            break
        targets.append(t)
        pb.confirm_reached()
    assert targets[0][0] > targets[-1][0]    # возврат идёт с конца к началу
    assert all(targets[i][0] > targets[i+1][0] for i in range(len(targets) - 1))

def test_simplify():
    pb = PathBuffer(min_distance=1.0)
    for i in range(50):
        pb.record((i, 0.001 * (i % 2)))      # почти прямая с шумом
    n_before = len(pb)
    pb.start_return()
    print(f"  simplify: {n_before} -> {len(pb)}")
    assert len(pb) < n_before // 2           # прямая сильно сократилась

def test_overflow():
    pb = PathBuffer(min_distance=0.1, max_points=100)
    for i in range(1000):
        pb.record((i * 0.2, math.sin(i) * 5))
    assert len(pb) <= 100                    # прореживание сработало

def test_save_load():
    pb = PathBuffer()
    for i in range(5):
        pb.record((i, i))
    pb.save("test_path.pkl")
    pb2 = PathBuffer()
    assert pb2.load("test_path.pkl")
    assert list(pb.stack) == list(pb2.stack)
    os.remove("test_path.pkl")


if __name__ == "__main__":
    test_record_threshold(); print("OK  порог записи")
    test_return_order();     print("OK  порядок возврата")
    test_simplify();         print("OK  упрощение пути")
    test_overflow();         print("OK  переполнение")
    test_save_load();        print("OK  save/load")
    print("\nВсе тесты path_buffer пройдены")
