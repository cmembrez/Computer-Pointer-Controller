"""
This is a class for a Head Pose Detection model.
This is the "intermediate level" in our pipeline (on the same level, there is the facial_landmarks_detection.py).
Head Pose Estimation gets its INPUT from face_detection.py, a cropped face image/frame, and
OUTPUT head pose angles (for the gaze_estimation.py).

The outputs are Tait-Bryan angles with z-y'-x'' (intrinsic rotations),
referred as yaw - pitch - roll.

The head_pose_estimation class has four methods
    load_model()
    predict(image)
    check_model()
    preprocess_input(image)
    preprocess_output(outputs, image)
"""

from openvino.inference_engine import IENetwork, IECore
import cv2
import math
import time
from pathlib import Path

from utils.rotation3d import draw_3d_axes
from utils.log_helper import LogHelper


class HeadPoseEstimation:
    """
    Class for the Head Pose Estimation Model.
    """
    def __init__(self, model_source, model_name, model_precision, device='CPU', extensions=None, threshold=0.5):
        """
        Set instance variables.
        """
        self.loggers = LogHelper()

        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.model_source = model_source
        self.model_name = model_name
        self.model_precision = model_precision
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        self.input_name = None
        self.input_shape = None
        self.outputs_names = None
        self.outputs_shapes = None

        if self.extensions is None:
            self.extensions = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
            self.loggers.main.info("HeadPoseEstimation: no extensions provided by user. Trying to add {}".format(self.extensions))

    def load_model(self):
        """
        This method is for loading the model to the device specified by the user.
        If the model requires any Plugins, this is where they are loaded.
        """
        # load intermediate representation (IR) files into related class
        project_path = Path(__file__).parent.parent.resolve()
        model_path = str(project_path) + "/models/" + self.model_source + "/" + self.model_name + "/" + self.model_precision + "/"
        model_xml = model_path + self.model_name + ".xml"
        model_bin = model_path + self.model_name + ".bin"

        # load the Inference Engine (IE) API
        self.plugin = IECore()

        # Read the IR as IENetwork
        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            self.loggers.main.error("HeadPoseEstimation: could not initialize the network. "
                          "Please check path to model. Exclude extensions (.xml, .bin).")
            print("Ced's printing: ", e)
            exit(1)

        # Get names and shapes
        iter_inputs = iter(self.network.inputs)
        self.input_name = []
        self.input_shape = []
        for i in range(len(self.network.inputs)):
            self.input_name.append(next(iter_inputs))
            self.input_shape.append(self.network.inputs[self.input_name[i]].shape)

        iter_outputs = iter(self.network.outputs)
        self.outputs_names = []
        self.outputs_shapes = []
        for i in range(len(self.network.outputs)):
            self.outputs_names.append(next(iter_outputs))
            self.outputs_shapes.append(self.network.outputs[self.outputs_names[i]].shape)

        # Add any necessary extensions
        try:
            if "CPU" in self.device:
                self.loggers.main.info("HeadPoseEstimation: CPU extensions not added.")
                # self.plugin.add_extension(self.extensions, self.device)

            # Get supported layers of the network
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

            # GPU extensions
            if "GPU" in self.device:
                supported_layers.update(self.plugin.query_network(self.network), 'CPU')

            # Check unsupported layers
            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                self.loggers.main.warning("HeadPoseEstimation: there are unsupported layers: {}".format(unsupported_layers))
                self.loggers.main.warning("HeadPoseEstimation: please add existing extension to the IECore if possible.")
                exit(1)
        except Exception as e:
            self.loggers.main.error("HeadPoseEstimation: problem with extensions provided. Please re-check info provided.")
            print("Ced's printing: ", e)
            exit(1)

        # Load the IENetwork into the plugin
        self.net_plugin = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.net_plugin

    def predict(self, image):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """

        # Preprocess INPUT (image)
        start_time = time.time()
        image_input_preprocessed = self.preprocess_input(image)
        self.loggers.benchmark.info("HeadPose;preprocess_input_time;{}".format(time.time() - start_time))

        # Infer
        input_dict = {self.input_name[0]: image_input_preprocessed}

        start_time = time.time()
        results = self.net_plugin.infer(input_dict)
        self.loggers.benchmark.info("HeadPose;inference_time;{}".format(time.time() - start_time))

        # Preproces OUTPUT
        start_time = time.time()
        coords = self.preprocess_output(results)
        self.loggers.benchmark.info("HeadPose;preprocess_output_time;{}".format(time.time() - start_time))

        return coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        INPUT: image, a cropped face from the face_detection.py
        OUTPUT: preprocessed frame ready for inference into head pose estimation model
        """
        image_input_preprocessed = image.copy()
        try:
            image_input_preprocessed = cv2.resize(image, (self.input_shape[0][3], self.input_shape[0][2]))
            image_input_preprocessed = image_input_preprocessed.transpose((2, 0, 1))
            image_input_preprocessed = image_input_preprocessed.reshape(1, *image_input_preprocessed.shape)
        except Exception as e:
            self.loggers.main.error("HeadPoseEstimation.preprocess_input(): inputs not conform. "
                          "Current input's shape is {}.".format(self.input_shape))
            print("Ced's printing: ", e)
            exit(1)

        return image_input_preprocessed

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        INPUT: outputs variable represents the coordinates from inference results
        OUTPUT: the Head Pose angles, that will be send to / used by gaze_estimation.py
        """
        # [0]=PITCH ('angle_p_fc'), Y
        # [1]=ROLL ('angle_r_fc'), X
        # [2]=YAW ('angle_y_fc'), Z
        # => but we return the "industry standard Z-Y-X"
        self.coords_accepted = [outputs[self.outputs_names[2]][0][0],
                                outputs[self.outputs_names[0]][0][0],
                                outputs[self.outputs_names[1]][0][0]]

        return self.coords_accepted

    def draw_output_on_frame(self, input_frame, face_coords):
        """

        :param input_frame:
        :param face_coords:
        :return:
        """

        # print("coords y-x-z: {}".format(self.coords_accepted))
        length = 200
        x0 = int((face_coords[0][0] + face_coords[0][2]) / 2)
        y0 = int((face_coords[0][1] + face_coords[0][3]) / 2)
        head_output_frame = input_frame.copy()

        '''
        old way -- wrong??
        
        # YAW (Z-axis)
        yaw_xy = get_end_point([x0, y0], self.coords_accepted[0], length)
        cv2.arrowedLine(head_output_frame, (x0, y0), (yaw_xy[0], yaw_xy[1]), (0, 0, 255), thickness=2)

        # PITCH (Y-axis)
        pitch_xy = get_end_point([x0, y0], self.coords_accepted[1], length)
        cv2.arrowedLine(head_output_frame, (x0, y0), (pitch_xy[0], pitch_xy[1]), (255, 0, 0), thickness=2)

        # ROLL (X-axis)
        roll_xy = get_end_point([x0, y0], self.coords_accepted[2], length)
        cv2.arrowedLine(head_output_frame, (x0, y0), (roll_xy[0], roll_xy[1]), (0, 255, 0), thickness=2)
        '''

        '''
        still in working progress
        '''
        fov = 1 / math.tan(70/2)  # 50 degree angle?
        wd = 45  # working distance = 45mm

        # is the focal length in terms of pixels? or in terms of distance? latter case, need to multiply by scale
        focal_length = 25  # approximate the focal length as the distance from the lens to the image, in mm
        scale = 50  # scale factor relating pixels to distance

        head_output_frame = draw_3d_axes(head_output_frame, [x0, y0],
                                         self.coords_accepted[0],
                                         self.coords_accepted[1],
                                         self.coords_accepted[2],
                                         scale=scale, focal_length=focal_length)

        return head_output_frame
