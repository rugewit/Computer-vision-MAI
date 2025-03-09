import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Определить номер дороги без препятствий в конце пути.
    
    :param image: входное изображение
    :return: индекс дороги без препятствий или -1, если такой нет
    """
    # Преобразуем изображение в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Создаем маску для выделения препятствий (красного цвета)
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, red_lower, red_upper)
    # Находим контуры препятствий на изображении
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Определяем размеры изображения
    _, width, _ = image.shape

    # Определяем количество дорог (на одну больше, чем количество найденных препятствий)
    num_routes = len(contours) + 1

    # Рассчитываем ширину каждой дороги
    route_width = width // num_routes
    
    # Создаем список для хранения информации о наличии препятствий на каждой дороге
    obstacle_map = [False] * num_routes
    
    # Определяем, какие дороги содержат препятствия
    for cnt in contours:
        # Получаем координаты ограничивающего прямоугольника
        x, _, _, _ = cv2.boundingRect(cnt)
        # Определяем индекс дороги
        route_index = min(x // route_width, num_routes - 1)
        # Помечаем наличие препятствия
        obstacle_map[route_index] = True
    # Находим первую дорогу без препятствий и возвращаем её индекс
    return next((i for i, has_obstacle in enumerate(obstacle_map) if not has_obstacle), -1)
