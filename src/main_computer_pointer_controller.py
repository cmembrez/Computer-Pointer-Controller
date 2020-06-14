#!/bin/python

"""
This is the main file of the application Computer Pointer Controller.

0) we start by reading the arguments given by the user, if any.
1) "first level": using input_feeder.py, the input (webcam or video) will first go through the face_detection.py
2) "intermediate level": The fac_detection.py outputs will be feed into
    a) facial_landmarks_detection.py: which in turns will output left and right eyes
    b) head_pose_estimation.py: which in turn will output the head pose angles
3) "final level": The outputs from 2a and 2b will be feed into the last model gaze_estimation.py.
    The latter will output towards mouse_controler.py in order to modify the user's mouse pointer position.

There exists a subfolder ./models.
Using the Intel OpenVINO Model Downloader,
the models .bin and .xml files may be located in the subfolders ./models/intel/<model_name>/<model_precision>/
"""

import argparse
import cv2
import sys
import time

from src.face_detection import FaceDetection
from src.facial_landmarks_detection import FacialLandmarks
from src.head_pose_estimation import HeadPoseEstimation
from src.gaze_estimation import GazeEstimation
from src.input_feeder import InputFeeder
from src.mouse_controller import MouseController

from utils.log_helper import LogHelper


def get_args():
    """
    Gets the arguments from the command line
    :return: parsed arguments
    """
    # setup the parser
    parser = argparse.ArgumentParser(
        description="Computer Pointer Controller APP: without any args, the app infers from the webcam on the CPU "
                    "using FP32 Face model and FP32-INT8 models.")
    # helper descriptions
    itype_desc = "The type of input. 'video', 'image' or 'cam' for webcam. Default=cam"
    ipath_desc = "The path to the input file. Default=0, for webcam."
    d_desc = "The device name: CPU (default), GPU or MYRIAD (for VPU)."

    model_source_desc = "Represents a subfolder in the subfolder ./models where you store your models. Default='intel'."

    face_model_desc = "The name of the face detection model in the subfolder " \
                      "./models/<model_source>/<faceModel>/<facePrecision>/<faceModel>.extensions. " \
                      "Default=face-detection-adas-binary-0001 "
    face_precision_desc = "The model precision for face detection model (cf arg faceModel_desc). Default=FP32-INT1"

    facial_model_desc = "The name of the facial landmarks detection model in the subfolder " \
                        "./models/<model_source>/<facialModel>/<facialPrecision>/<facialModel_desc>.extensions. " \
                        "Default=landmarks-regression-retail-0009 "
    facial_precision_desc = "The model precision for facial landmarks detection model. Default=FP32"

    head_model_desc = "The path to the head pose estimation model in the subfolder " \
                      "./models/<model_source>/<headModel>/<headPrecision>/<headModel>.extensions. " \
                      "Default=head-pose-estimation-adas-0001 "
    head_precision_desc = "The model precision for head pose estimation model. Default=FP32"

    gaze_model_desc = "The path to the gaze estimation model in the subfolder " \
                      "./models/<model_source>/<gazeModel>/<gazePrecision>/<gazeModel>.extensions. " \
                      "Default=gaze-estimation-adas-0002 "
    gaze_precision_desc = "The model precision for gaze estimation model. Default=FP32"

    gui_desc = "Set to True to activate the graphic user interface (GUI). Default=False."

    # create the arguments
    parser.add_argument("-itype", "--inputtype", help=itype_desc, default='cam', required=False)
    parser.add_argument("-ipath", "--inputpath", help=ipath_desc, default=0, required=False)
    parser.add_argument("-d", "--device", help=d_desc, default="CPU", required=False)

    parser.add_argument("-modelSource", help=model_source_desc, default="intel", required=False)

    parser.add_argument("-faceModel", help=face_model_desc, default="face-detection-adas-binary-0001", required=False)
    parser.add_argument("-facePrecision", help=face_precision_desc, default="FP32-INT1", required=False)

    parser.add_argument("-facialModel", help=facial_model_desc, default="landmarks-regression-retail-0009",
                        required=False)
    parser.add_argument("-facialPrecision", help=facial_precision_desc, default="FP32", required=False)

    parser.add_argument("-headModel", help=head_model_desc, default="head-pose-estimation-adas-0001", required=False)
    parser.add_argument("-headPrecision", help=head_precision_desc, default="FP32", required=False)

    parser.add_argument("-gazeModel", help=gaze_model_desc, default="gaze-estimation-adas-0002", required=False)
    parser.add_argument("-gazePrecision", help=gaze_precision_desc, default="FP32", required=False)

    parser.add_argument("-g", "--gui", help=gui_desc, default=False, required=False)

    args = parser.parse_args()

    return args


def main():
    # This will initiate two loggers that write to ../log/main.log, and /benchmark.log
    loggers = LogHelper()
    # Arguments from user
    args = get_args()

    # Initialize model classes
    model_face = FaceDetection(model_source=args.modelSource, model_name=args.faceModel,
                               model_precision=args.facePrecision, device=args.device, extensions=None, threshold=0.5)
    model_facial = FacialLandmarks(model_source=args.modelSource, model_name=args.facialModel,
                                   model_precision=args.facialPrecision, device=args.device, extensions=None,
                                   threshold=0.5)
    model_head = HeadPoseEstimation(model_source=args.modelSource, model_name=args.headModel,
                                    model_precision=args.headPrecision, device=args.device, extensions=None,
                                    threshold=0.5)
    model_gaze = GazeEstimation(model_source=args.modelSource, model_name=args.gazeModel,
                                model_precision=args.gazePrecision, device=args.device, extensions=None,
                                threshold=0.5)

    mouse_precision = 'high'
    mouse_speed = 'fast'
    mouse_controller = MouseController(mouse_precision, mouse_speed)

    # TIMER
    # model loading time
    # input/output processing
    # model inference time
    # log the model name and precision
    loggers.benchmark.info("Face;{};{}".format(args.faceModel, args.facePrecision))
    loggers.benchmark.info("Facial;{};{}".format(args.facialModel, args.facialPrecision))
    loggers.benchmark.info("Head;{};{}".format(args.headModel, args.headPrecision))
    loggers.benchmark.info("Gaze;{};{}".format(args.gazeModel, args.gazePrecision))

    # Load models
    start_time = time.time()
    model_face.load_model()
    loggers.benchmark.info("Face;model_load_time;{}".format(time.time()-start_time))

    start_time = time.time()  # reset time for next model
    model_facial.load_model()
    loggers.benchmark.info("FacialLandmarks;model_load_time;{}".format(time.time()-start_time))

    start_time = time.time()
    model_head.load_model()
    loggers.benchmark.info("HeadPose;model_load_time;{}".format(time.time()-start_time))

    start_time = time.time()
    model_gaze.load_model()
    loggers.benchmark.info("Gaze;model_load_time;{}".format(time.time()-start_time))

    # read input
    feed = InputFeeder(input_type=args.inputtype, input_file=args.inputpath)
    feed.load_data()
    if not (feed.cap.isOpened()):
        loggers.main.critical("Could not open input device ({}). Exiting now...".format(feed.input_type))
        exit(1)

    width = int(feed.cap.get(3))
    height = int(feed.cap.get(4))

    for batch in feed.next_batch():
        if batch is None:
            break
        '''
        INFERENCE
        '''
        # FACE
        batch_face = batch.copy()
        face_coords, face_image_cropped = model_face.predict(batch_face)

        # FACIAL LANDMARKS
        if (face_image_cropped is None) or (len(face_image_cropped) == 0):
            loggers.main.info("FaceDetection: no detection above threshold.")
        else:
            batch_facial = batch.copy()
            facial_coords, image_eyes = model_facial.predict(batch_facial, face_image_cropped)

            # HEAD POSE
            head_angles = model_head.predict(face_image_cropped)

            # GAZE
            if (len(head_angles) == 0) or (len(image_eyes) == 0):
                loggers.main.info("HeadPoseEstimation: no detection above threshold.")
            else:
                gaze_results = model_gaze.predict(image_eyes, head_angles)

                # MOUSE CONTROL
                try:
                    mouse_controller.move(gaze_results[0][0], gaze_results[0][1])
                except KeyboardInterrupt:
                    loggers.main.info("main_computer_pointer_controller: exit on mouse controller move!")
                    sys.exit()



            '''
            SHOW OUTPUTS TO USER: uncomment the desired section (i.e. remove ''' ''') 
            '''
            # face
            '''
            cv2.imshow("original", batch)
            cv2.imshow("face only", batch_face)
            '''

            # facial landmarks
            '''
            batch_facial = batch.copy()
            face_facial_image = model_facial.draw_output_on_frame(batch_face, face_image_cropped, face_coords)
            cv2.imshow("face + facial landmarks", face_facial_image)
            only_facial_image = model_facial.draw_output_on_frame(batch_facial, face_image_cropped, face_coords)
            cv2.imshow("only facial landmarks", only_facial_image)
            full_facial_image = model_facial.draw_nose_lips_on_frame(only_facial_image, face_image_cropped, face_coords)
            cv2.imshow("all facial landmarks", full_facial_image)
            '''

            # head pose
            '''
            batch_head = batch.copy()
            only_head_image = model_head.draw_output_on_frame(batch_head, face_coords)
            cv2.imshow("only head pose", only_head_image)
            '''

            # gaze
            '''
            batch_gaze = batch.copy()
            only_gaze_image = model_gaze.draw_output_on_frame(batch_gaze, facial_coords, face_coords, face_image_cropped)
            cv2.imshow("only gaze", only_gaze_image)
            '''

        if cv2.waitKey(60) == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    feed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
