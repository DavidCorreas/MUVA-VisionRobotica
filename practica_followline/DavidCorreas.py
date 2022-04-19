from GUI import GUI
from HAL import HAL
import cv2.cv2
import numpy as np

print(np.__version__)

TURN = 2.6
GRADUAL_HORIZON = 0.0000001
SHARP_TURN = 60
KP = 0.35
KD = 0.65
MAX_R = MAX_L = 0
HORIZON = 200


# Create mask
def turn(x):
    if x > 0:
        return - (x ** 2)
    else:
        return x ** 2


def process_line():
    # Enter iterative code!
    img = HAL.getImage()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get Line
    mask1 = cv2.inRange(img, (0, 50, 20), (15, 255, 255))
    mask2 = cv2.inRange(img, (160, 50, 20), (180, 255, 255))
    raw_line_mask = cv2.bitwise_or(mask1, mask2) / 255

    kernel = np.ones((7, 7), np.uint8)
    erode_line_mask = cv2.erode(raw_line_mask, kernel)
    line_mask = cv2.dilate(erode_line_mask, kernel)

    return img, line_mask


def calculate_heatmap(shape):
    h, w, _ = shape
    turn_err_row = np.array(
        [turn(xi) for xi in np.arange((TURN * -1), TURN, (2 * TURN) / w)]).astype('float64')

    turn_err = turn_err_row.copy()
    for i in range(h - 1):
        gradual_sum = (np.ones((1, w)) * (GRADUAL_HORIZON * i))
        gradual_sum[:, w // 2:] *= -1
        turn_err = np.vstack((turn_err_row + gradual_sum, turn_err))
    # turn_err = np.tile(turn_err_row, (h, 1))
    turn_err[-HORIZON:, :] = 0

    return turn_err


# Enter sequential code!
speed = 5
HAL.setV(speed)
img, line_mask = process_line()
heatmap = calculate_heatmap(img.shape)

while True:
    img, line_mask = process_line()

    # Line with turn
    line_turn_err = line_mask * heatmap

    # No line found
    while len(np.nonzero(line_turn_err)[0]) <= 0:
        HAL.setV(0)
        speed = 3
        if 'pid_turn_value' not in locals():
            pid_turn_value = 2
        HAL.setW(pid_turn_value)
        img, line_mask = process_line()
        line_turn_err = line_mask * heatmap
        normalized_im = abs(line_turn_err * 150).astype(np.uint8)
        therm_im_line_turn_err = cv2.applyColorMap(normalized_im, cv2.COLORMAP_JET)

        GUI.showImage(therm_im_line_turn_err)

    # == turn == #
    raw_turn_value = line_turn_err.sum() / np.count_nonzero(line_turn_err)

    if 'old_raw_turn_value' not in locals():
        old_raw_turn_value = raw_turn_value

    diff_turn_value = (raw_turn_value - old_raw_turn_value)
    pid_turn_value = KP * raw_turn_value + KD * diff_turn_value

    if 'old_diff_turn_value' not in locals():
        old_diff_turn_value = diff_turn_value
    if 'oscilation' not in locals():
        oscilation = 1
    if 'confidence' not in locals():
        confidence = 0

    # Recta
    if abs(pid_turn_value) < 0.1:
        confidence = 0.1
        KP = 0.35
        KD = 0.65

    # Curva brusca
    elif np.count_nonzero(line_turn_err[:, :SHARP_TURN]) > 0 or np.count_nonzero(line_turn_err[:, -SHARP_TURN:]) > 0:
        print('Curva brusca')
        speed = 5
        confidence = 0
        KP = 0.50
        KD = 0.50

    # Curva
    else:
        KP = 0.35
        KD = 0.65
        # From Right to Left
        if old_diff_turn_value > 0 and diff_turn_value < 0:
            MAX_R = raw_turn_value
            oscilation = MAX_R - MAX_L

        if old_diff_turn_value < 0 and diff_turn_value > 0:
            MAX_L = raw_turn_value
            oscilation = MAX_R - MAX_L

        if oscilation < 0.1:
            confidence = 0.05
        elif oscilation < 0.3:
            confidence = 0.05
        elif oscilation < 0.8:
            confidence = 0.01
        elif oscilation < 1.5:
            confidence = 0
        else:
            print('En curva')
            confidence = 0
    speed = confidence + speed

    old_diff_turn_value = diff_turn_value
    old_raw_turn_value = raw_turn_value

    HAL.setV(speed)
    HAL.setW(pid_turn_value)


    # == print == #
    normalized_im = abs(line_turn_err * 150).astype(np.uint8)
    therm_im_line_turn_err = cv2.applyColorMap(normalized_im, cv2.COLORMAP_JET)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    GUI.showImage(therm_im_line_turn_err)
