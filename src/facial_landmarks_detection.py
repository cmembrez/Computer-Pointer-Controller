"""
This is a class for a Facial Landmarks Detection model.
This is the "intermediate level" in our pipeline (on the same level, there is the head_pose_estimation.py).
Facial Landmarks Detection gets its INPUT from face_detection.py, a cropped face image/frame, and
OUTPUT cropped Left & Right Eyes (for the gaze_estimation.py).

The facial_landmarks class has four methods
    load_model()
    predict(image)
    check_model()
    preprocess_input(image)
    preprocess_output(outputs, image)
"""

from openvino.inference_engine import IENetwork, IECore
import cv2
import time
from pathlib import Path

from utils.tools_image import crop_square_from_point
from utils.log_helper import LogHelper

class FacialLandmarks:
    """
    Class for the Facial Landmarks Detection Model.
    """
    def __init__(self, model_source, model_name, model_precision, device='CPU', extensions=None, threshold=0.5):
        """
        Setting instance variables.
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
        self.output_name = None
        self.output_shape = None

        self.outputs_detections = None

        if self.extensions is None:
            self.extensions = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64" \
                              "/libcpu_extension_sse4.so "
            self.loggers.main.info("FacialLandmarks: no extensions provided by user. Trying to add {}".format(self.extensions))

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
            self.loggers.main.error("FacialLandmarks: could not initialize the network. "
                          "Please check path to model. Exclude extensions (.xml, .bin).")
            print("Ced's printing: ", e)
            exit(1)

        # Get names and shapes
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

        # Add any necessary extensions
        try:
            if "CPU" in self.device:
                self.loggers.main.info("FacialLandmarks: CPU extensions not added.")
                # self.plugin.add_extension(self.extensions, self.device)

            # Get supported layers of the network
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

            # GPU extensions
            if "GPU" in self.device:
                supported_layers.update(self.plugin.query_network(self.network), 'CPU')

            # Check unsupported layers
            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                self.loggers.main.warning("FacialLandmarks: there are unsupported layers: {}".format(unsupported_layers))
                self.loggers.main.warning("FacialLandmarks: please add existing extension to the IECore if possible.")
                exit(1)
        except Exception as e:
            self.loggers.main.error("FacialLandmarks: problem with extensions provided. Please re-check info provided.")
            print("Ced's printing: ", e)
            exit(1)

        # Load the IENetwork into the plugin
        self.net_plugin = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.net_plugin

    def predict(self, face_image_full, face_image_cropped):
        """
        This method is meant for running predictions on the input image.
        """
        # Preprocess INPUT (image)
        start_time = time.time()
        image_input_preprocessed = self.preprocess_input(face_image_cropped)
        self.loggers.benchmark.info("FacialLandmarks;preprocess_input_time;{}".format(time.time() - start_time))
        # Infer
        input_dict = {self.input_name: image_input_preprocessed}

        start_time = time.time()
        results = self.net_plugin.infer(input_dict)
        self.loggers.benchmark.info("FacialLandmarks;inference_time;{}".format(time.time() - start_time))

        self.outputs_detections = results[self.output_name]
        # Preproces OUTPUT
        start_time = time.time()
        image_eyes = self.preprocess_output(face_image_full)
        self.loggers.benchmark.info("FacialLandmarks;preprocess_output_time;{}".format(time.time() - start_time))

        #return self.outputs_detections[0][0:4], image_eyes
        return self.outputs_detections[0], image_eyes

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        INPUT: image, a frame from a webcam or a video file
        OUTPUT: preprocessed frame ready for inference into FACE DETECTION MODEL
        """
        image_input_preprocessed = image.copy()
        try:
            image_input_preprocessed = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image_input_preprocessed = image_input_preprocessed.transpose((2, 0, 1))
            image_input_preprocessed = image_input_preprocessed.reshape(1, *image_input_preprocessed.shape)
        except Exception as e:
            self.loggers.main.error("FacialLandmarks.preprocess_input(): inputs not conform. "
                          "Current input's shape is {}.".format(self.input_shape))
            print("Ced's printing: ", e)
            exit(1)

        return image_input_preprocessed

    def preprocess_output(self, face_image_full):
        """
        The model predicts five facial landmarks: 2x eyes, 1x nose, 2x lip corners

        INPUT: a blob of shape [1, 10] = a row vector of 10 floating point
                for five landmarks:= (x0,y0, x1,y1, ..., x5,y5)
        OUTPUT: we preprocessed our outputs to have cropped left and right eyes
                for the gaze_estimation.py (our 'final level' in the pipeline).
        """

        next_model_input_shape = 60  # from openVINO documentation on Gaze model

        image_left_eye = crop_square_from_point(face_image_full,
                                                [self.outputs_detections[0][0], self.outputs_detections[0][1]],
                                                square_reshape_size=60)
        image_right_eye = crop_square_from_point(face_image_full,
                                                 [self.outputs_detections[0][2], self.outputs_detections[0][3]],
                                                 square_reshape_size=60)

        return [image_left_eye, image_right_eye]

    def draw_output_on_frame(self, input_frame, face_image_cropped, face_coords):
        """
        Return the frame with added/drawn outputs (circles for left & right eyes) in it
        :param face_coords: coordinates (x0,y0) and (x1,y1) for the detected Face (top-left, bottom-right corners)
        :param face_image_cropped: a cropped frame representing the detected Face
        :param input_frame: the original batch/frame to draw on it
        :return: same-size frame with output drawn on it.
        """
        face_facial_image = input_frame.copy()
        try:
            global_left_eye_x = int(self.outputs_detections[0][0] * face_image_cropped.shape[0]) + \
                                    face_coords[0][0]
            global_left_eye_y = int(self.outputs_detections[0][1] * face_image_cropped.shape[1]) + \
                                    face_coords[0][1]
            global_right_eye_x = int(self.outputs_detections[0][2] * face_image_cropped.shape[0]) + \
                                    face_coords[0][0]
            global_right_eye_y = int(self.outputs_detections[0][3] * face_image_cropped.shape[1]) + \
                                    face_coords[0][1]

            face_facial_image = cv2.circle(face_facial_image, center=(global_left_eye_x, global_left_eye_y),
                                           radius=5, color=(0, 0, 255), thickness=2)
            face_facial_image = cv2.circle(face_facial_image, center=(global_right_eye_x, global_right_eye_y),
                                           radius=5, color=(0, 0, 255), thickness=2)

        except Exception as e:
            self.loggers.main.error("FacialLandkmarks.draw_output_on_frame: ", e)

        return face_facial_image


    def draw_nose_lips_on_frame(self, input_frame, face_image_cropped, face_coords):
        """
        Return the frame with added/drawn outputs (circles for nose and left & right lip corners) in it
        :param face_coords: coordinates (x0,y0) and (x1,y1) for the detected Face (top-left, bottom-right corners)
        :param face_image_cropped: a cropped frame representing the detected Face
        :param input_frame: the original batch/frame to draw on it
        :return: same-size frame with output drawn on it.
        """
        face_facial_image = input_frame.copy()
        try:
            nose_x = int(self.outputs_detections[0][4] * face_image_cropped.shape[0]) + \
                         face_coords[0][0]
            nose_y = int(self.outputs_detections[0][5] * face_image_cropped.shape[1]) + \
                         face_coords[0][1]
            left_lip_x = int(self.outputs_detections[0][6] * face_image_cropped.shape[0]) + \
                            face_coords[0][0]
            left_lip_y = int(self.outputs_detections[0][7] * face_image_cropped.shape[1]) + \
                            face_coords[0][1]
            right_lip_x = int(self.outputs_detections[0][8] * face_image_cropped.shape[0]) + \
                            face_coords[0][0]
            right_lip_y = int(self.outputs_detections[0][9] * face_image_cropped.shape[1]) + \
                            face_coords[0][1]
            face_facial_image = cv2.circle(face_facial_image, center=(nose_x, nose_y),
                                           radius=5, color=(0, 0, 255), thickness=1)
            face_facial_image = cv2.circle(face_facial_image, center=(left_lip_x, left_lip_y),
                                           radius=3, color=(0, 0, 255), thickness=1)
            face_facial_image = cv2.circle(face_facial_image, center=(right_lip_x, right_lip_y),
                                           radius=3, color=(0, 0, 255), thickness=1)

        except Exception as e:
            self.loggers.main.error("FacialLandkmarks.draw_nose_lips_on_frame: ", e)

        return face_facial_image
