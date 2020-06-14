"""
Helper functions related to image manipulation: cropping, etc.

crop_square_from_point(): will return a squared cropped image around a specific point.
get_end_point(): return a pair (x1,y1) that is "length" away from its starting point (x0,y0), based on an "angle".
"""

import logging
import cv2
import numpy as np
import math
# from IntelEdgeAI_IoTDeveloper.starter.utils.log_helper import LogHelper


def crop_square_from_point(image, square_center, square_reshape_size=60):
    """
    Helper function to crop a squared frame around the detected 'eye' coordinate (x,y). It will serves as input for gaze_estimation.py
    :param image: input to be cropped around square_center
    :param square_center: a (x,y) referring to the center of the output cropped zone, in range [0,1]
    :param square_reshape_size: the side in pixel of the square
    :return: a squared cropped image with side length = square_reshape_size
    """
    # loggers = LogHelper()
    # loggers.main.debug("tools_image: square center: {}".format(square_center))
    # # loggers.main.debug("tools_image: image shape: {}".format(image.shape))

    x_min = square_center[0] * image.shape[0]
    y_min = square_center[1] * image.shape[1]

    image_x_min = int(x_min - square_reshape_size / 2)
    image_y_min = int(y_min - square_reshape_size / 2)
    image_x_max = int(image_x_min + square_reshape_size)
    image_y_max = int(image_y_min + square_reshape_size)

    # # loggers.main.debug("tools_image: x_min ,y_min: {}, {}".format(image_x_min, image_y_min))
    # # loggers.main.debug("tools_image: x_max, y_max: {}, {}".format(image_x_max, image_y_max))

    image_eye = image.copy()
    image_eye = image_eye[image_x_min:image_x_max,
                image_y_min:image_y_max]

    # # loggers.main.debug("tools_image: image out shape: {}".format(image_eye.shape))

    return image_eye


def get_end_point(start_point, angle_in_degree, length):
    """
    Given a starting point (x0, y0), an angle in degree and a length, it returns the second point coordinate (x1, y1)
    :param start_point: (x0, y0) coordinates of the starting point
    :param angle: in degree
    :param length: length of desired distance between start and end points.
    :return: a pair (x1, y1) representing the end point.
    """
    angle0_radian = angle_in_degree * np.pi / 180

    x1 = int(start_point[0] + length * math.cos(angle0_radian))
    y1 = int(start_point[1] + length * math.sin(angle0_radian))

    return [x1, y1]
