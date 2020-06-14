"""
This is a class for a Face Detection model.
This is the "first level" in our pipeline: it reads INPUT from a webcam or video file (input_feeder.py),
and ultimately OUTPUT a cropped face to the next two models (facial_landmarks_detection.py, and head_pose_estimation.py).

The face_detection class has five methods
    load_model()
    predict(image)
    check_model()
    preprocess_input(image)
    preprocess_output(outputs, image)
"""
import openvino
from openvino.inference_engine import IENetwork, IECore
import cv2
import time
from pathlib import Path

from utils.log_helper import LogHelper


class FaceDetection:
    """
    Class for the Face Detection Model.
    """
    def __init__(self, model_source, model_name, model_precision, device='CPU', extensions=None, threshold=0.5):
        """
        Set Instance Variables
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

    def load_model(self):
        """
        This method is for loading the model to the device specified by the user.
        If the model requires any Plugins, this is where they are loaded.
        :return:
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
            self.loggers.main.error("FaceDetection: could not initialize the network. Please check path to model. Exclude extensions (.xml, .bin).")
            print("Ced's printing: ", e)
            exit(1)

        # Get names and shapes
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

        # Add any necessary extensions
        self.check_model()

        # Load the IENetwork into the plugin
        self.net_plugin = self.plugin.load_network(self.network, self.device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.net_plugin

    def predict(self, image):
        """
        Run prediction on the input image
        :param image:
        :return:
        """
        # Preprocess INPUT (image)
        start_time = time.time()
        image_input_preprocessed = self.preprocess_input(image)
        self.loggers.benchmark.info("Face;preprocess_input_time;{}".format(time.time() - start_time))
        # Infer
        input_dict = {self.input_name:image_input_preprocessed}

        start_time = time.time()
        results = self.net_plugin.infer(input_dict)
        self.loggers.benchmark.info("Face;inference_time;{}".format(time.time() - start_time))

        outputs_detections = results[self.output_name]

        # Preproces OUTPUT
        start_time = time.time()
        coords, image_face_cropped = self.preprocess_output(outputs_detections, image)
        self.loggers.benchmark.info("Face;preprocess_output_time;{}".format(time.time() - start_time))

        return coords, image_face_cropped

    def check_model(self):
        """
        Checking the model for unsupported layers. and checking if openvino version < 2020 for CPU Extensions needs.
        :return: void. info is in main.log
        """
        try:
            '''
            openvino_version = int(openvino.__file__.split("/")[3].split("_")[1].split(".")[0])
            self.loggers.main.info("OPENVINO: user is using a version >= {}".format(openvino_version))

            if ("CPU" in self.device) and (openvino_version < 2020) and (self.extensions is None):
                self.extensions = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
                self.plugin.add_extension(self.extensions, self.device)
                self.loggers.main.info(
                    "FaceDetection: no extensions provided by user. Added {}".format(self.extensions))
            '''
            # Get supported layers of the network
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)

            # GPU extensions
            '''
            if "GPU" in self.device:
                supported_layers.update(self.plugin.query_network(self.network), 'CPU')
            '''

            # Check unsupported layers
            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                self.loggers.main.warning("FaceDetection: there are unsupported layers: {}".format(unsupported_layers))
                self.loggers.main.warning("FaceDetection: please add existing extension to the IECore if possible.")
                exit(1)
        except Exception as e:
            self.loggers.main.error("FaceDetection: problem with extensions provided. Please re-check info provided.")
            print("Ced's printing: ", e)
            exit(1)

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        INPUT: image, a frame from a webcam or a video file
        OUTPUT: preprocessed frame ready for inference into FACE DETECTION MODEL
        :param image:
        :return:
        """
        image_input_preprocessed = image.copy()
        try:
            image_input_preprocessed = cv2.resize(image_input_preprocessed, (self.input_shape[3], self.input_shape[2]))
            image_input_preprocessed = image_input_preprocessed.transpose((2, 0, 1))
            image_input_preprocessed = image_input_preprocessed.reshape(1, *image_input_preprocessed.shape)
        except Exception as e:
            self.loggers.main.error("FaceDetection.preprocess_input(): inputs not conform. "
                          "Current input's shape is {}.".format(self.input_shape))
            print("Ced's printing: ", e)
            exit(1)

        return image_input_preprocessed

    def preprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        INPUT: outputs variable represents the coordinates from inference results
        OUTPUT: coordinates of the face-detected, full image with rectangle-detection,
        and a cropped face that will be sent to the next model, FACIAL LANDMARK DETECTION
        :param outputs:
        :param image:
        :return:
        """
        coords_accepted = [] # coordinates actually given threshold
        image_cropped = None

        for boxdetection in outputs[0][0]:
            confidence_result = boxdetection[2]
            if confidence_result >= self.threshold:
                xmin = int(boxdetection[3] * image.shape[1]) # * image width
                ymin = int(boxdetection[4] * image.shape[0]) # * image height
                xmax = int(boxdetection[5] * image.shape[1])
                ymax = int(boxdetection[6] * image.shape[0])

                # add coordinates to be returned
                coords_accepted.append((xmin, ymin, xmax, ymax))

                # draw a box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                image_cropped = image.copy()
                image_cropped = image_cropped[xmin:(xmax+1), ymin:(ymax+1)]

        return coords_accepted, image_cropped
