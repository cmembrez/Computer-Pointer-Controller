'''
This is a class for a Gaze Estimation model.
This is the "final level" in our pipeline.
It received INPUTs from: facial_landmarks_detection.py (left/right eyes) and head_pose_estimation.py (head pose angles).
It will OUTPUT a 3-D vector corresponding to the user's gaze.
This output goes toward the mouse_controller.py to impact user's mouse pointer position.

The head_pose_estimation class has five methods
    load_model()
    predict(image)
    check_model()
    preprocess_input(image)
    preprocess_output(outputs, image)
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import time
from pathlib import Path
from utils.log_helper import LogHelper

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_source, model_name, model_precision, device='CPU', extensions=None, threshold=0.5):
        '''
        Set instance variables.
        '''
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

        self.inputs_names = None
        self.inputs_shapes = None
        self.outputs_names = None
        self.outputs_shapes = None
        self.outputs_coords = None

        if self.extensions == None:
            self.extensions = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
            self.loggers.main.info("GazeEstimation: no extensions provided by user. Trying to add {}".format(self.extensions))

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
            self.loggers.main.error("GazeEstimation: could not initialize the network. "
                          "Please check path to model. Exclude extensions (.xml, .bin).")
            print("Ced's printing: ", e)
            exit(1)

        # Get names and shapes
        iter_inputs = iter(self.network.inputs)
        self.inputs_names = []
        self.inputs_shapes = []
        for i in range(len(self.network.inputs)):
            self.inputs_names.append(next(iter_inputs))
            self.inputs_shapes.append(self.network.inputs[self.inputs_names[i]].shape)

        iter_outputs = iter(self.network.outputs)
        self.outputs_names = []
        self.outputs_shapes = []
        for i in range(len(self.network.outputs)):
            self.outputs_names.append(next(iter_outputs))
            self.outputs_shapes.append(self.network.outputs[self.outputs_names[i]].shape)

        # Add any necessary extensions
        try:
            if "CPU" in self.device:
                self.loggers.main.info("GazeEstimation: CPU extensions not added.")
                # self.plugin.add_extension(self.extensions, self.device)

            # Get supported layers of the network
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

            # GPU extensions
            if "GPU" in self.device:
                supported_layers.update(self.plugin.query_network(self.network), 'CPU')

            # Check unsupported layers
            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                self.loggers.main.warning("GazeEstimation: there are unsupported layers: {}".format(unsupported_layers))
                self.loggers.main.warning("GazeEstimation: please add existing extension to the IECore if possible.")
                exit(1)
        except Exception as e:
            self.loggers.main.error("GazeEstimation: problem with extensions provided. Please re-check info provided.")
            print("Ced's printing: ", e)
            exit(1)
        # Load the IENetwork into the plugin
        self.net_plugin = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.net_plugin

    def predict(self, image_eyes, head_angles):
        """
        This method is meant for running predictions on the input image.
        :param image_eyes: a size-two vector with image_eyes[0] representing cropped image eye of shape (60, 60, 3)
        :param head_angles: a vector with the three angles (yaw, pitch, roll) from head_pose_estimation
        :return: a right-handed coordinate (x,y,z) where x is orthogonal, y vertical
                and z-axis is directed from person's eyes to camera center
        """

        # Preprocess INPUT (image)
        start_time = time.time()
        image_input_preprocessed = self.preprocess_input(image_eyes)
        self.loggers.benchmark.info("Gaze;preprocess_input_time;{}".format(time.time() - start_time))
        # Infer
        input_dict = {"head_pose_angles": head_angles, "left_eye_image": image_input_preprocessed[0],
                      "right_eye_image": image_input_preprocessed[1]}

        start_time = time.time()
        results = self.net_plugin.infer(input_dict)
        self.loggers.benchmark.info("Gaze;inference_time;{}".format(time.time() - start_time))

        outputs_detections = results[self.outputs_names[0]]
        # Preproces OUTPUT
        start_time = time.time()
        self.outputs_coords = self.preprocess_output(outputs_detections)
        self.loggers.benchmark.info("Gaze;preprocess_output_time;{}".format(time.time() - start_time))

        return self.outputs_coords

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
            image_input_preprocessed[0] = cv2.resize(image_input_preprocessed[0],
                                                     (self.inputs_shapes[1][3], self.inputs_shapes[1][2]))
            image_input_preprocessed[0] = image_input_preprocessed[0].transpose((2, 0, 1))
            image_input_preprocessed[0] = image_input_preprocessed[0].reshape(1, *image_input_preprocessed[0].shape)

            image_input_preprocessed[1] = cv2.resize(image_input_preprocessed[1],
                                                     (self.inputs_shapes[2][3], self.inputs_shapes[2][2]))
            image_input_preprocessed[1] = image_input_preprocessed[1].transpose((2, 0, 1))
            image_input_preprocessed[1] = image_input_preprocessed[1].reshape(1, *image_input_preprocessed[1].shape)
        except Exception as e:
            self.loggers.main.error("GazeEstimation.preprocess_input(): inputs not conform. "
                          "Current input's shape is {}.".format(self.inputs_shapes))
            print("Ced's printing: ", e)

            self.loggers.main.debug("Inputs for Gaze estimation not available.")
            self.loggers.main.debug("image[0].shape: {}".format(image_input_preprocessed[0].shape))
            self.loggers.main.debug("image[1].shape: {}".format(image_input_preprocessed[1].shape))
            self.loggers.main.debug("inputs_shapes.shape: {}".format(self.inputs_shapes))

            exit(1)

        return image_input_preprocessed

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        INPUT: outputs variable represents the coordinates from inference results
        OUTPUT: the Head Pose angles, that will be send to / used by gaze_estimation.py
        """
        # main_loggerdebug("GazeEstimation.preprocess_output(): WHY NOTHING IS DONE IN THAT METHOD???")
        return outputs

    def draw_output_on_frame(self, face_facial_image, facial_coords, face_coords, face_image_cropped):
        """

        :param face_image_cropped: a cropped frame representing the detected Face
        :param input_frame: the original batch/frame to draw on it
        :param face_coords: coordinates (x0,y0) and (x1,y1) for the detected Face (top-left, bottom-right corners)
        :return:
        """
        return_image = face_facial_image.copy()
        # mid-point between left & right eyes' center
        global_left_eye_x = int(facial_coords[0] * face_image_cropped.shape[0]) + \
                            face_coords[0][0]
        global_left_eye_y = int(facial_coords[1] * face_image_cropped.shape[1]) + \
                            face_coords[0][1]
        global_right_eye_x = int(facial_coords[2] * face_image_cropped.shape[0]) + \
                             face_coords[0][0]
        global_right_eye_y = int(facial_coords[3] * face_image_cropped.shape[1]) + \
                             face_coords[0][1]

        x_mid = int((global_left_eye_x + global_right_eye_x) / 2)
        y_mid = int((global_left_eye_y + global_right_eye_y) / 2)

        # Arbitrarily Set Arrow Length: to a fourth of the smallest side of original frame
        length = int(min(return_image.shape[0:2]) / 4)

        # not normalized, non-unit length: self.outputs_coords  __ x, y, z
        # they start from (x_mid, y_mid)
        # z-axis goes into camera direction | y-axis = vertical | x-axis = horizontal
        cv2.arrowedLine(return_image, (x_mid, y_mid), (int(length * self.outputs_coords[0][0] + x_mid),
                                                       int(length * self.outputs_coords[0][1] + y_mid)), (0, 0, 255),
                        thickness=5)

        return return_image
