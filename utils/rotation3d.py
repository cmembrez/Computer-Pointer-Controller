"""
3d rotation - Tait-Bryan angles
Code based on Udacity inputs
Adjusted with mathworks and wiki sources
"""

import logging
import cv2
import numpy as np
import math


def draw_3d_axes(frame, center_of_face, yaw_degree, pitch_degree, roll_degree, scale, focal_length):
    """

    :param frame:
    :param center_of_face:
    :param yaw_degree: in degree
    :param pitch_degree: in degree
    :param roll_degree: in degree
    :param scale: to create X-Y-Z basic matrices
    :param focal_length: passed to camera_build_matrix()
    :return: frame with added line and circle on Z-axis
    """
    # from degree to radian: recall 180° constitutes np.pi radians
    yaw_radian = yaw_degree * np.pi / 180.0
    pitch_radian = pitch_degree * np.pi / 180.0
    roll_radian = roll_degree * np.pi / 180.0

    # Origin coordinates: get the center of the detected face
    origin_x = int(center_of_face[0])
    cy = int(center_of_face[1])

    '''
    Euler (Tait-Bryan) Angles to Rotation Matrix
    '''
    rotation_matrix = euler_angles_to_rotation_matrix(yaw_radian, pitch_radian, roll_radian)
    intrinsic_matrix = create_intrinsic_matrix([origin_x, cy], focal_length)

    # X-Y-Z axis: input
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    zero_matrix = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    zero_matrix[2] = intrinsic_matrix[0][0]

    # X-Y-Z axis: applying rotation
    xaxis = np.dot(rotation_matrix, xaxis) + zero_matrix
    yaxis = np.dot(rotation_matrix, yaxis) + zero_matrix
    zaxis = np.dot(rotation_matrix, zaxis) + zero_matrix
    zaxis1 = np.dot(rotation_matrix, zaxis1) + zero_matrix

    print("np.dot(rotation_matrix, xaxis) + zero_matrix: {}".format(np.dot(rotation_matrix, xaxis) + zero_matrix))
    print("np.dot(rotation_matrix, xaxis): {}".format(np.dot(rotation_matrix, xaxis)))

    '''
    Draw Axis: screen projection
    
    BlueGreenRed
    '''

    # X-axis: draw line RED
    xp2 = (xaxis[0] / xaxis[2] * intrinsic_matrix[0][0]) + origin_x
    yp2 = (xaxis[1] / xaxis[2] * intrinsic_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (origin_x, cy), p2, (0, 0, 255), 2)

    # Y-axis: draw line GREEN
    xp2 = (yaxis[0] / yaxis[2] * intrinsic_matrix[0][0]) + origin_x
    yp2 = (yaxis[1] / yaxis[2] * intrinsic_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (origin_x, cy), p2, (0, 255, 0), 2)

    # Z-axis: get points 1 and 2 BLUE
    xp1 = (zaxis1[0] / zaxis1[2] * intrinsic_matrix[0][0]) + origin_x
    yp1 = (zaxis1[1] / zaxis1[2] * intrinsic_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))

    xp2 = (zaxis[0] / zaxis[2] * intrinsic_matrix[0][0]) + origin_x
    yp2 = (zaxis[1] / zaxis[2] * intrinsic_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))

    # Z-axis: draw a line + a circle
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)

    return frame


def euler_angles_to_rotation_matrix(yaw_radian, pitch_radian, roll_radian):
    """
    Given Tait-Bryan angles yaw (z), pitch (y) and roll (x), return a rotation matrix
    (ref: en.wikipedia.org/wiki/Rotation_matrix)
    :param yaw_radian:
    :param pitch_radian:
    :param roll_radian:
    :return: a 3x3 rotation matrix
    """
    # 3D Basic Rotation Matrices
    basic_z = np.array([[math.cos(yaw_radian),      0,                          -math.sin(yaw_radian)],
                        [0,                         1,                          0],
                        [math.sin(yaw_radian),      0,                          math.cos(yaw_radian)]])

    basic_y = np.array([[1,                         0,                          0],
                        [0,                         math.cos(pitch_radian),     -math.sin(pitch_radian)],
                        [0,                         math.sin(pitch_radian),     math.cos(pitch_radian)]])

    basic_x = np.array([[math.cos(roll_radian),     -math.sin(roll_radian),     0],
                        [math.sin(roll_radian),     math.cos(roll_radian),      0],
                        [0,                         0,                          1]])

    # 3D General Rotation: Intrinsic rotation with Tait-Bryan angles
    # matrix multiplication "@"
    general_rotation = basic_z @ basic_y @ basic_x

    return general_rotation


def create_intrinsic_matrix(optical_center, focal_length, skew_coefficient=0):
    """
    Return the intrinsic matrix to map the camera coordinates into the image plane.
    (ref: mathworks.com/help/vision/ug/camera-calibration.html)
    (ref: en.wikipedia.org/wiki/Camera_matrix, 'intrinsic parameters only)
    (ref: en.wikipedia.org/wiki/Camera_resectioning)
    :param skew_coefficient: for completeness only! set to zero here
    :param principal_point_offset: origin point of our axes = the x and y coordinates of the detected face
    :param focal_length: (i.e. the distance between the pinhole and the film/image plane)
    :return: 3x3 intrinsic matrix
    """
    #  create the matrix with the intrinsic parameters
    intrinsic_matrix = np.array([[focal_length,   skew_coefficient,     optical_center[0]],
                              [0,                 focal_length,         optical_center[1]],
                              [0,                 0,                    1]])

    return intrinsic_matrix
