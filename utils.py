"""
Utils for the project
"""
import cv2

from common import *


def get_probable_eyes(eyes_list):
    """
    Find the two closest areas in size (eyes are usually similar in size)
    """

    eye_sizes = []

    for _, _, w, h in eyes_list:
        eye_sizes.append(w * h)

    if len(eyes_list) > 1:
        diffs = []
        for i in range(len(eye_sizes) - 1):
            min_indexes = None
            min_diff = None
            for j in range(i + 1, len(eye_sizes)):
                diff = abs(eye_sizes[i] - eye_sizes[j])
                if not min_diff or diff < min_diff:
                    min_diff = diff
                    min_indexes = (i, j)

            diffs.append((min_indexes, min_diff))

        indexes, _ = diffs[min(range(len(diffs)), key=lambda i: diffs[i][1])]
        return [eyes_list[indexes[0]], eyes_list[indexes[1]]]
    else:
        return eyes_list


def place_greenscreen_image(bg, fg, top_left_x, top_left_y):
    """
    Place image fg (removing greenscreen) on background image bg
    """
    fg_hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)

    for pixel_row_num in range(len(fg)):
        for pixel_col_num in range(len(fg[pixel_row_num])):
            pixel = fg_hsv[pixel_row_num, pixel_col_num]
            if not (BG_MID_BOUND + SENSITIVITY) > pixel[0] > (BG_MID_BOUND - SENSITIVITY):
                bg[top_left_y + pixel_row_num, top_left_x + pixel_col_num] = fg[pixel_row_num, pixel_col_num]

    return bg


def resize_image(image, new_height=0, new_width=0):
    """
    Resize the image using the new_width and/or new_height
    """
    assert new_height or new_width

    if not new_height:
        new_height = round(image.shape[0] * (new_width / image.shape[1]))
    elif not new_width:
        new_width = round((new_height / image.shape[0]) * image.shape[1])

    resized_image = cv2.resize(image, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)
    return resized_image
