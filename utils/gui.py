"""
Despite the name, the current class only outputs tables of results.
"""

import numpy as np
from pathlib import Path

class GuiPointerController:
    def __init__(self):
        # SHOW OUTPUT / GUI
        self.window = None

    def analyze_benchmark_log(self):
        """
        try to get most info out of benchmark.log
        :return: void()  # print statements so far
        """
        project_path = Path(__file__).parent.parent.resolve()
        data_log = np.loadtxt(str(project_path) + "/log/benchmark.log", delimiter=";", usecols=(2, 3, 4), dtype='str')
        data_log = np.char.lstrip(data_log)

        face_data = data_log[data_log[:, 0] == 'Face', 1:]
        face_load = face_data[face_data[:, 0] == 'model_load_time', 1:][0][0].astype(np.float).round(4)
        face_input = face_data[face_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        face_output = face_data[face_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        face_inference = face_data[face_data[:, 0] == 'inference_time', 1:].astype(np.float)

        facial_data = data_log[data_log[:, 0] == 'FacialLandmarks', 1:]
        facial_load = facial_data[facial_data[:, 0] == 'model_load_time', 1:][0][0].astype(np.float).round(4)
        facial_input = facial_data[facial_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        facial_output = facial_data[facial_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        facial_inference = facial_data[facial_data[:, 0] == 'inference_time', 1:].astype(np.float)

        head_data = data_log[data_log[:, 0] == 'HeadPose', 1:]
        head_load = head_data[head_data[:, 0] == 'model_load_time', 1:][0][0].astype(np.float).round(4)
        head_input = head_data[head_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        head_output = head_data[head_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        head_inference = head_data[head_data[:, 0] == 'inference_time', 1:].astype(np.float)

        gaze_data = data_log[data_log[:, 0] == 'Gaze', 1:]
        gaze_load = gaze_data[gaze_data[:, 0] == 'model_load_time', 1:][0][0].astype(np.float).round(4)
        gaze_input = gaze_data[gaze_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        gaze_output = gaze_data[gaze_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        gaze_inference = gaze_data[gaze_data[:, 0] == 'inference_time', 1:].astype(np.float)

        print("(seconds)| load time \t | avg preprocess input \t | inference time \t | avg preprocess output")
        print("face:\t | {} \t\t | {} \t\t\t\t\t | {} \t\t\t | {}".format(face_load, np.mean(face_input).round(4), np.mean(face_inference).round(4),
                                      np.mean(face_output).round(4)))
        print("facial:\t | {} \t\t | {} \t\t\t\t\t | {} \t\t\t | {}".format(facial_load, np.mean(facial_input).round(4), np.mean(facial_inference).round(4),
                                      np.mean(facial_output).round(4)))
        print("head:\t | {} \t\t | {} \t\t\t\t\t | {} \t\t\t | {}".format(head_load, np.mean(head_input).round(4), np.mean(head_inference).round(4),
                                      np.mean(head_output).round(4)))
        print("gaze:\t | {} \t\t | {} \t\t\t\t\t | {} \t\t\t | {}".format(gaze_load, np.mean(gaze_input).round(4), np.mean(gaze_inference).round(4),
                                      np.mean(gaze_output).round(4)))
        print("-"*100)


def main():
    basic_gui = GuiPointerController()
    basic_gui.analyze_benchmark_log()


if __name__ == '__main__':
    main()
