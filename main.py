# import time
import numpy as np


def generate_points(N, min_x, max_x, min_y, max_y):
    x_values = np.random.uniform(min_x, max_x, N)
    y_values = np.random.uniform(min_y, max_y, N)
    return np.array((x_values, y_values, np.zeros(N))).T


def generate_points2(sx, sy, robot, n):
    w = 1.0 / n
    weights = np.full(n, w)
    return np.array([np.random.uniform(robot[0] - sx, robot[0] + sx, n),
                     np.random.uniform(robot[1] - sy, robot[1] + sy, n),
                     weights]).T


def find_distance(point, k, lidar_dist, room):
    # Обчислення кутів для вимірювань лідара
    angles = (2.0 * np.pi / k) * np.arange(k)

    # Генерація точок сканування на основі відстані лідара, кутів та вказаної точки
    scan_points = np.column_stack((lidar_dist * np.cos(angles) + point[0],
                                   lidar_dist * np.sin(angles) + point[1]))

    # Перетворення координат кімнати в масив NumPy для зручності маніпуляцій
    walls = np.array(room)

    # Зсув кінців стін для створення векторів, які представляють стіни
    wall_endpoints = np.roll(walls, -1, axis=0)
    wall_vectors = wall_endpoints - walls

    # Розширення розмірності точок сканування, векторів стін та стін для передачі
    scan_points_expanded = np.expand_dims(scan_points, axis=1)
    wall_vectors_expanded = np.expand_dims(wall_vectors, axis=0)
    walls_expanded = np.expand_dims(walls, axis=0)

    # Обчислення відносних координат точок сканування щодо початкових точок стін
    relative_to_start = scan_points_expanded - walls_expanded

    # Обчислення скалярних добутків між векторами стін та відносними координатами
    dot_product_ab_ap = np.sum(wall_vectors_expanded * relative_to_start, axis=2)

    # Обчислення квадратних величин векторів стін
    squared_magnitude_ab = np.sum(wall_vectors_expanded ** 2, axis=2)

    # Обчислення відстані від точок сканування до ліній, визначених стінами
    distance_to_line = np.abs(wall_vectors_expanded[:, :, 0] * relative_to_start[:, :, 1]
                              - relative_to_start[:, :, 0] * wall_vectors_expanded[:, :, 1]) / np.sqrt(
        squared_magnitude_ab)

    # Обчислення відстані до кінцевих точок на основі скалярних добутків
    dist_to_endpoints = np.where(dot_product_ab_ap < 0, np.sqrt(np.sum(relative_to_start ** 2, axis=2)),
                                 np.where(dot_product_ab_ap >= squared_magnitude_ab, np.linalg.norm(relative_to_start -
                                                                                wall_vectors_expanded, axis=2),
                                          distance_to_line))

    # Пошук мінімальних відстаней до кінцевих точок для кожної точки сканування
    min_distances = np.min(dist_to_endpoints, axis=1)

    # Повернення середньої мінімальної відстані для всіх точок сканування
    return np.sum(min_distances ** 2) / k


def calc_weights(particle, k, sl, lidar_data, room_point):
    sum_w = 0
    res = []
    for i in particle:
        distance = find_distance([i[0], i[1]], k, lidar_data, room_point)
        normalization_factor = 1 / (sl * np.sqrt(2 * np.pi))
        exponential_factor = np.exp(-0.5 * ((distance ** 0.5) ** 2) / (sl ** 2))
        i[2] = normalization_factor * exponential_factor
        sum_w += normalization_factor * exponential_factor
    for i in particle:
        i[2] /= sum_w
        res.append(i[2])
    return res


def repopulation(weights, points):
    indexs = []
    index = np.random.randint(0, len(points))
    betta = 0
    for i in range(len(points)):
        betta += np.random.uniform(0, 2 * max(weights))
        while betta > weights[index]:
            betta -= weights[index]
            index = (index + 1) % len(points)
        indexs.append(index)
    return points[indexs]


def solve(points, k, sl, logs, room, move_point, sx, sy):
    sum_q = 0
    for index, i in enumerate(logs):
        weight = calc_weights(points, k, sl, i, room)
        for q in points:
            sum_q += q[2]
        for q in points:
            q[2] /= sum_q
        sum_q = 0
        points = repopulation(weight, points)
        for q in points:
            sum_q += q[2]
        for q in points:
            q[2] /= sum_q
        if index != len(logs) - 1:
            move = np.array([move_point[index][0], move_point[index][1], 0])
            points = points + move
            points += np.hstack([np.random.normal(scale=sx, size=(len(points), 1)),
                                 np.random.normal(scale=sy, size=(len(points), 1)), np.zeros((len(points), 1))])
    x, y = 0, 0
    for i in range(len(points)):
        x += points[i][0] * points[i][2]
        y += points[i][1] * points[i][2]
    print(x, y)


if __name__ == '__main__':
    n = int(input())
    room_points = np.array(list(map(float, input().split()))).reshape(-1, 2)
    min_x, max_x = min(room_points[:, 0]), max(room_points[:, 0])
    min_y, max_y = min(room_points[:, 1]), max(room_points[:, 1])
    room_points = [[point[0], point[1]] for point in room_points]
    m, k = map(int, input().split())
    sl, sx, sy = map(float, input().split())
    flag = input()
    if flag.split()[0] != '0':
        x0, y0 = list(map(float, flag.split()))[1:]
        particles = generate_points2(sx, sy, [x0, y0], 200)
    else:
        particles = generate_points(200, min_x, max_x, min_y, max_y)

    dist = [np.array(list(map(float, input().split())))]
    move = []
    for i in range(m):
        move.append(list(map(float, input().split())))
        dist.append(list(map(float, input().split())))
    # s = time.time()
    dist = np.array(dist)
    move = np.array(move)
    solve(particles, k, sl, dist, room_points, move, sx, sy)
