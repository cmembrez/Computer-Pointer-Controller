'''
a Graphical-User-Interface to output webcam with inferences' results and statistics

With additional buttons and optionality.

'''

import tkinter as tk
import PIL.Image, PIL.ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GuiPointerController:
    def __init__(self):
        # SHOW OUTPUT / GUI
        window = tk.Tk()
        cv_img = cv2.imread("../bin/forestPiano.jpeg")
        height, width, no_channels = cv_img.shape

        canvas = tk.Canvas(window, width=width, height=height)
        canvas.pack()

        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        btn_close = tk.Button(window, text="Close Window", width=50, command=window.destroy)
        btn_close.pack(anchor=tk.CENTER, expand=True)

        # window.mainloop()

    def analyze_benchmark_log(self):
        """
        try to get most info out of benchmark.log
        :return: void()  # print statements so far
        """
        data_log = np.loadtxt("../log/benchmark.log", delimiter=";", usecols=(2, 3, 4), dtype='str')
        data_log = np.char.lstrip(data_log)

        face_data = data_log[data_log[:, 0] == 'Face', 1:]
        face_load = face_data[face_data[:, 0] == 'model_load_time', 1:].astype(np.float)
        face_input = face_data[face_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        face_output = face_data[face_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        face_inference = face_data[face_data[:, 0] == 'inference_time', 1:].astype(np.float)

        facial_data = data_log[data_log[:, 0] == 'FacialLandmarks', 1:]
        facial_load = facial_data[facial_data[:, 0] == 'model_load_time', 1:].astype(np.float)
        facial_input = facial_data[facial_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        facial_output = facial_data[facial_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        facial_inference = facial_data[facial_data[:, 0] == 'inference_time', 1:].astype(np.float)

        head_data = data_log[data_log[:, 0] == 'HeadPose', 1:]
        head_load = head_data[head_data[:, 0] == 'model_load_time', 1:].astype(np.float)
        head_input = head_data[head_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        head_output = head_data[head_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        head_inference = head_data[head_data[:, 0] == 'inference_time', 1:].astype(np.float)

        gaze_data = data_log[data_log[:, 0] == 'Gaze', 1:]
        gaze_load = gaze_data[gaze_data[:, 0] == 'model_load_time', 1:].astype(np.float)
        gaze_input = gaze_data[gaze_data[:, 0] == 'preprocess_input_time', 1:].astype(np.float)
        gaze_output = gaze_data[gaze_data[:, 0] == 'preprocess_output_time', 1:].astype(np.float)
        gaze_inference = gaze_data[gaze_data[:, 0] == 'inference_time', 1:].astype(np.float)

        print("load time")
        print("face: {}".format(face_load))
        print("facial: {}".format(facial_load))
        print("head: {}".format(head_load))
        print("gaze: {}".format(gaze_load))

        print("----------------------")
        print("avg preprocess input")
        print("face: {}".format(sum(face_input)/len(face_inference)))
        print("facial: {}".format(sum(facial_input)/len(facial_inference)))
        print("head: {}".format(sum(head_input) / len(head_inference)))
        print("gaze: {}".format(sum(gaze_input) / len(gaze_inference)))

        print("----------------------")
        print("avg inference")
        print("face: {}".format(sum(face_inference)/len(face_inference)))
        print("facial: {}".format(sum(facial_inference)/len(facial_inference)))
        print("head: {}".format(sum(head_inference) / len(head_inference)))
        print("gaze: {}".format(sum(gaze_inference) / len(gaze_inference)))

        print("----------------------")
        print("avg preprocess output")
        print("face: {}".format(sum(face_output)/len(face_inference)))
        print("facial: {}".format(sum(facial_output)/len(facial_inference)))
        print("head: {}".format(sum(head_output) / len(head_inference)))
        print("gaze: {}".format(sum(gaze_output) / len(gaze_inference)))

        print("----------------------")


        '''
        plt.plot(face_inference)
        plt.plot(face_output)
        plt.plot(face_input)
        plt.show()
        '''


def main():
    basicGui = GuiPointerController()

    basicGui.analyze_benchmark_log()


if __name__ == '__main__':
    main()
