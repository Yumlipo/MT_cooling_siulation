import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from itertools import groupby

# === Функция для расчёта "катастроф" (резких спадов) в данных ===
def catastrophe_calc(lengths, fig_name):
    """
    lengths  : массив значений (например, длины микротрубочек во времени)
    fig_name : имя файла для сохранения графика (png/jpg)

    Функция:
      - сглаживает данные (savgol_filter)
      - вычисляет производную (градиент)
      - находит участки с резким спадом (slope < threshold)
      - отбрасывает слишком короткие флуктуации
      - строит график с выделенными зонами
      - сохраняет результат в файл
      - возвращает количество найденных "катастроф"
    """

    # === 1. Сглаживание данных ===
    window_size = 500                # размер окна сглаживания (должен быть нечётным и < len(lengths))
    smoothed = savgol_filter(lengths, window_size, 3)  # сглаживание полиномом 3-й степени

    # === 2. Вычисляем первую производную (градиент) ===
    slope = np.gradient(smoothed)    # показывает скорость изменения

    # === 3. Задаём порог для обнаружения тренда ===
    # increase_threshold = 0.15       # (не используется, только для роста)
    decrease_threshold = -1.1         # если скорость < -1.1 → резкое падение ("катастрофа")

    # === 4. Классификация трендов ===
    trend = np.zeros_like(slope)     # 0 — без изменений
    # trend[slope > increase_threshold] = 1   # можно включить для роста
    trend[slope < decrease_threshold] = -1   # падение (красный)

    # === 5. Поиск непрерывных сегментов тренда ===
    segments = []
    for key, group in groupby(enumerate(trend), key=lambda x: x[1]):  # группируем по значению (-1, 0, 1)
        if key != 0:   # пропускаем нейтральные сегменты
            indices = [i for i, val in group]
            start, end = indices[0], indices[-1]
            segments.append((start, end, key))   # (начало, конец, тип тренда)

            # Отладочные принты
            print("start", start)
            print("end", end)

    # === 6. Фильтрация коротких сегментов ===
    min_segment_length = 3
    filtered_segments = [
        (s, e, t) for s, e, t in segments if (e - s + 1) >= min_segment_length
    ]

    # === 7. Построение графика ===
    plt.figure(figsize=(10, 6))

    # 1. Исходные и сглаженные данные
    plt.plot(lengths, 'o-', linewidth=3, label='Original Data', alpha=0.5)
    plt.plot(smoothed, 'b-', linewidth=2, label='Smoothed Data')

    # 2. Закрашиваем сегменты с трендом
    colors = {1: 'green', -1: 'red'}  # рост = зелёный, падение = красный
    for start, end, trend_dir in filtered_segments:
        plt.axvspan(start, end, alpha=0.2, color=colors[trend_dir], 
                    label=f'{"Increasing" if trend_dir == 1 else "Decreasing"} Trend')

    # 3. Строим график производной (на второй оси)
    plt.twinx()
    plt.plot(slope, 'k--', label='Slope (Derivative)')
    # plt.axhline(increase_threshold, color='green', linestyle=':', alpha=0.5, label='Increase Threshold')
    plt.axhline(decrease_threshold, color='red', linestyle=':', alpha=0.5, label='Decrease Threshold')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # === 8. Подписи и легенда ===
    plt.title('Trend Detection in Data (Ignoring Small Fluctuations)')
    plt.xlabel('Index')
    plt.ylabel('Value / Slope')
    plt.legend(loc='upper left')
    plt.grid(True)

    # === 9. Сохраняем картинку ===
    plt.savefig(fig_name)

    # plt.show()   # можно включить для интерактивного просмотра

    return len(filtered_segments)   # возвращаем число катастроф
