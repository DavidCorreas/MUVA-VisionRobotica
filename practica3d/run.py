# from GUI import GUI
# from HAL import HAL
import cv2.cv2 as cv
import numpy as np


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
    :return: line equation in 3D space given direction and point.
    """
    cam_3d_esc = HAL.getCameraPosition(position_cam)
    optical_point_2d = HAL.graficToOptical(position_cam, [*image_point_2d, 1])
    direction_vect = HAL.backproject(position_cam, optical_point_2d)[:3]


# Get points
# im_left = HAL.getImage('left')
# im_right = HAL.getImage('right')

im_left = cv.imread('img_left.png')
im_right = cv.imread('img_left.png')

points_left_2d = get_interesting_points(im_left)

for point in points_left_2d:
    cv.circle(im_left, point, 1, (0, 0, 255), -1)









cv.imshow('left', im_left)
cv.waitKey(0)



# while True:
#
#     cv.imshow("Image right", im_right)
#     cv.imshow("Image left", im_left)
#     cv.waitKey(0)
    # GUI.showImages(im_left, im_right, True)
