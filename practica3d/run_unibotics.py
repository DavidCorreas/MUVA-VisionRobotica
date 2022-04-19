from GUI import GUI
from HAL import HAL
import cv2.cv2 as cv
import numpy as np
import random


def get_interesting_points(img) -> list:
    """
    Get the interesting points of the image
    :param img: image to be processed
    :return: list of interesting points (i, j)
    """
    canny = cv.Canny(img, 100, 200)
    points = [(x, y) for y, x in zip(*canny.nonzero()) if y < img.shape[1] // 2]
    print(f"{len(points)} points found")
    return points


def get_3d_line_equation(position_cam: str, image_point_2d: tuple) -> tuple:
    """
    Get the line equation in 3D space
    :param position_cam: position of the camera ('right' or 'left')
    :param image_point: point of the image
    :return: line equation in 3D space given direction and point. Both in homogeneous (P3).
    """
    cam_3d_esc = np.append(HAL.getCameraPosition(position_cam), [1])
    optical_point_2d = HAL.graficToOptical(position_cam, [*image_point_2d, 1])
    direction_vect = HAL.backproject(position_cam, optical_point_2d)
    return cam_3d_esc, np.array(direction_vect - cam_3d_esc)


def get_epipolar_line(position_cam: str, opposite_ray: tuple, epipolar_thickness: int=9) -> np.array:
    """
    Get the epipolar line in 3D space given the opposite ray
    :param position_cam: position of the camera ('right' or 'left')
    :param opposite_ray: ray of the opposite camera (direction, point)
    :return: mask of the epipolar line on image of `opposite_ray`
    """
    # Project two points on position_cam image
    p1_3d = opposite_ray[0] + opposite_ray[1]
    p1_2d_position_cam = HAL.project(position_cam, p1_3d)

    # Point 5 times farther than p1
    p2_3d = (5 * opposite_ray[0]) + opposite_ray[1]
    p2_2d_position_cam = HAL.project(position_cam, p2_3d)
    # print(f'p1_2d: {p1_2d_position_cam}, p2_2d: {p2_2d_position_cam}')

    p1_graph = HAL.opticalToGrafic(position_cam, p1_2d_position_cam).astype(np.int)
    p2_graph = HAL.opticalToGrafic(position_cam, p2_2d_position_cam).astype(np.int)

    def line_intersection(x1, y1, x2, y2, x):
        return (y2 - y1) * (x - x1) / (x2 - x1) + y1

    # Get the intersection of the line with the image
    img = HAL.getImage(position_cam)
    left_x_edge, right_x_edge = 0, img.shape[1]
    left_y_intersection = line_intersection(p1_graph[0], p1_graph[1], p2_graph[0], p2_graph[1], left_x_edge)
    right_y_intersection = line_intersection(p1_graph[0], p1_graph[1], p2_graph[0], p2_graph[1], right_x_edge)

    p0 = np.array([left_x_edge, left_y_intersection]).astype(np.int)
    p1 = np.array([right_x_edge, right_y_intersection]).astype(np.int)

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv.line(mask, tuple(p0), tuple(p1), 255, epipolar_thickness)
    masked_img = cv.bitwise_and(img, img, mask=mask)
    return masked_img


def get_homologue_point(masked_img, left_img, left_point_2d):
    """
    Find the homologue point in the opposite image given the point in the masked image.
    :param masked_img: masked image
    :param opposite_img: opposite image
    :param point_2d: point in the opposite image to find in masked image
    :return:
    """
    # Match the region of point_2d in the masked_img image
    opposite_region = left_img[left_point_2d[1] - 2:left_point_2d[1] + 2, left_point_2d[0] - 2:left_point_2d[0] + 2]
    point_right_2d = cv.matchTemplate(masked_img, opposite_region, cv.TM_CCOEFF_NORMED)
    point_right_2d = np.unravel_index(np.argmax(point_right_2d), point_right_2d.shape)
    return point_right_2d[1], point_right_2d[0]


def get_middle_point_between_lines(vector_right, vector_left, camera_right, camera_left):
    # Get middle point between the two lines
    # Create the system Ax = b to solve with the least squares method
    n = np.cross(vector_left, vector_right)
    A = np.array([vector_left, n, -vector_right]).T
    b = camera_right - camera_left
    # Solve the system
    alpha, beta, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    point3d = (alpha * vector_left) + ((beta / 2) * n)
    print(f'point3d: {point3d}')
    return HAL.project3DScene(point3d).tolist()


GUI.ClearAllPoints()

im_left = HAL.getImage('left')
im_right = HAL.getImage('right')
GUI.showImages(im_left, im_right, True)


# ========== 1. GET INTESRESTING POINTS ==========
points_left_2d = get_interesting_points(im_left)
points2d = random.sample(points_left_2d, 5000)

point_struct = []
for point_left_2d in points2d:
    # ===== DEBUG ======
    cv.circle(im_left, point_left_2d, 3, (0, 255, 0), -1)
    struct = {
        'm_left': {
            'x': point_left_2d[0],
            'y': point_left_2d[1]
        }
    }

    # ========== 2. GET LEFT RAY ==========
    point_left_3d, direction_left_3d = get_3d_line_equation('left', point_left_2d)
    struct['ray_left'] = {
            'point': point_left_3d,
            'direction': direction_left_3d}

    # ========== 3. GET LINE MASK ==========
    masked = get_epipolar_line('right', (struct['ray_left']['direction'], struct['ray_left']['point']))
    struct['masked_line'] = masked

    # ========== 4. GET HOMOLOGUE POINT ==========
    im_left_raw = HAL.getImage('left')
    point_right_2d = get_homologue_point(struct['masked_line'], im_left_raw, point_left_2d)
    cv.circle(im_right, point_right_2d, 3, (0, 255, 0), -1)
    # GUI.showImageMatching(point_left_2d[0], point_left_2d[1], point_right_2d[0], point_right_2d[1])

    # ========== 4. TRIANGULATE ==========
    point_right_3d, direction_right_3d = get_3d_line_equation('right', point_right_2d)
    struct['ray_right'] = {
        'point': point_right_3d,
        'direction': direction_right_3d}

    point_3d = get_middle_point_between_lines(struct['ray_right']['direction'][:3], struct['ray_left']['direction'][:3],
                                              struct['ray_right']['point'][:3], struct['ray_left']['point'][:3])

    print(f'point_3d: {point_3d}')

    GUI.ShowNewPoints([point_3d + [255, 0, 0]])

    # ========== DEBUG. PRINT IMAGE ==========
    GUI.showImages(im_left, masked, True)


while True:
    pass
