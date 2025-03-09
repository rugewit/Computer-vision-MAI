import cv2
import numpy as np


def rotate(image, point, angle):
    # Получаем размеры изображения (высота и ширина)
    (h, w) = image.shape[:2]
    
    # Получаем матрицу поворота для заданной точки и угла
    M = cv2.getRotationMatrix2D(point, angle, scale=1.0)

    # Определяем координаты углов изображения (включая смещение для однородных координат)
    last_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    last_corners = np.hstack([last_corners, np.ones((last_corners.shape[0], 1))])
    # Применяем матрицу трансформации к углам изображения
    new_corners = M @ last_corners.T

    # Находим минимальные и максимальные координаты новых углов
    min_x = np.min(new_corners[0, :])
    max_x = np.max(new_corners[0, :])
    min_y = np.min(new_corners[1, :])
    max_y = np.max(new_corners[1, :])

    # Вычисляем новые размеры изображения после поворота
    new_w = int(np.round(max_x - min_x))
    new_h = int(np.round(max_y - min_y))

    # Корректируем смещение в матрице поворота, чтобы изображение поместилось
    M[0, 2] -= min_x
    M[1, 2] -= min_y

    # Применяем трансформацию и возвращаем результат
    return cv2.warpAffine(image, M, (new_w, new_h))


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    (h, w) = image.shape[:2]
    affine_matrix = cv2.getAffineTransform(points1, points2)

    # Определяем координаты углов изображения (включая смещение для однородных координат)
    last_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    last_corners = np.hstack([last_corners, np.ones((last_corners.shape[0], 1))])
    # Применяем матрицу трансформации к углам изображения
    new_corners =  affine_matrix @ last_corners.T

    # Находим минимальные и максимальные координаты новых углов
    min_x = np.min(new_corners[0, :])
    max_x = np.max(new_corners[0, :])
    min_y = np.min(new_corners[1, :])
    max_y = np.max(new_corners[1, :])

    # Вычисляем новые размеры изображения после поворота
    new_w = int(np.round(max_x - min_x))
    new_h = int(np.round(max_y - min_y))

    # Корректируем смещение в матрице поворота, чтобы изображение поместилось
    affine_matrix[0, 2] -= min_x
    affine_matrix[1, 2] -= min_y

    return cv2.warpAffine(image, affine_matrix, (new_w, new_h))
