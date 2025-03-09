import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def find_way_from_maze(image: np.ndarray) -> tuple:
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Бинаризация изображения (инвертируем, чтобы стены стали черными, а проходы белыми)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    h, w = binary.shape  # Получаем размеры изображения
    
    # Определяем возможные точки входа (верхний край) и выхода (нижний край)
    start = [(0, x) for x in range(w) if binary[0, x] == 255]
    end = [(h-1, x) for x in range(w) if binary[h-1, x] == 255]

    start = start[0]
    end = end[0]
    
    # Если нет входа или выхода, возвращаем None
    if not start or not end:
        return None
    
    # Используем очередь для BFS (поиск в ширину)
    queue = deque([start])
    visited = set([start])  # Множество посещенных клеток
    prev = dict()
    prev[start] = None
    
    # Возможные направления движения (вверх, вниз, влево, вправо)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (y, x) = queue.popleft()  # Извлекаем текущую позицию и путь
        
        # Если дошли до выхода
        if (y, x) == end:
            break
        
        # Перебираем все возможные направления движения
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            # Проверяем, что новая позиция в пределах изображения и является проходом
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited and binary[ny, nx] == 255:
                queue.append((ny, nx))
                visited.add((ny, nx))
                prev[(ny, nx)] = (y, x)
    
    path = []
    cur_step = end
    while cur_step:
        path.append(cur_step)
        cur_step = prev[cur_step]
    x_all, y_all = zip(*path)
    return np.array(x_all), np.array(y_all)