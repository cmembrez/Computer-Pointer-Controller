# Computer Pointer Controller

The current application will estimate the gaze of the user, and the gaze's coordinates will be used to move the position of the user's mouse.

To estimate the gaze of the user, the application reads an input (webcam, video) and uses an inference pipeline of four pre-trained models.

## Project Set Up and Installation
* You will need to have Intel OpenVINO toolkit installed (version >= 2020.1), please refer to their official [link]( https://docs.openvinotoolkit.org/latest/index.html)

* You can start by cloning the current respository: git clone www.github.com/cmembrez/computer_pointer_controller
* See below section on 'how to install the dependencies'.
* Place four models in the ./models folder. You can download pre-trained models from Intel OpenVINO resources.
* Finally, in a terminal, run the bash script test_simple.sh.

Please refer to the './log' folder for logging information. Please note that they are reset at each new launch:
* benchmark.log: gives for each model used, its name, its precision, its loading time, and each batch's time on 
input/output preprocessing and inference time.
* layers.log: gives a detailed performance analysis per layer for each model (limited to one analysis per model).
* main.log: gives any important event that occured related to the application as a whole (e.g. missing model, unsupported layer, etc.)
* results.txt: is a summary of several tests on a local CPU.

If you are curious, feel free to check the PDFs for additional overview of the project through diagrams.

#### project directory structure

<pre>
├── bin
│   └── demo.mp4
├── log
│   ├── benchmark.log
│   ├── layers.log
│   ├── main.log
│   └── results.txt
├── models
│   └── model_source
│       └── model_name
│           └── model_precision
│               ├── model_name.bin
│               └── model_name.xml
├── README.md
├── requirements.txt
├── src
│   ├── face_detection.py
│   ├── facial_landmarks_detection.py
│   ├── gaze_estimation.py
│   ├── head_pose_estimation.py
│   ├── input_feeder.py
│   ├── main_computer_pointer_controller.py
│   ├── model.py
│   └── mouse_controller.py
└── utils
    ├── gui.py
    ├── log_helper.py
    ├── rotation3d.py
    └── tools_image.py
</pre>

#### models you need to download 
The current application has been developed with four models:
* [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

#### where to place them
Please refer also to the section "Command Line Arguments Explained" to know how to place models' files with flags.
If you download four models face-detection-adas-binary-0001, gaze-estimation-adas-0002,
head-pose-estimation-adas-0001, landmarks-regression-retail-0009, with the Intel OpenVINO downloader into ./models,
you would have a similar structure in the models folder:

<pre>
.
└── models
    └── intel
        ├── face-detection-adas-binary-0001
        │   └── FP32-INT1
        │       ├── face-detection-adas-binary-0001.bin
        │       └── face-detection-adas-binary-0001.xml
        ├── gaze-estimation-adas-0002
        │   ├── FP16
        │   │   ├── gaze-estimation-adas-0002.bin
        │   │   └── gaze-estimation-adas-0002.xml
        │   ├── FP32
        │   │   ├── gaze-estimation-adas-0002.bin
        │   │   └── gaze-estimation-adas-0002.xml
        │   └── FP32-INT8
        │       ├── gaze-estimation-adas-0002.bin
        │       └── gaze-estimation-adas-0002.xml
</pre>

#### how to install the dependencies
This application has been developed in a Python 3.6 virtual environment. 
Please refer to the requirements.txt in the root folder for a list of the dependencies required, and to your own 
environment's documentation for specific instructions to install them. 
As an example, PyCharm makes it easy to install via its plugin, or in your (virtual) environment, 
you can run a command similar to:
 
<pre>
$ pip3 install -r requirements.txt
</pre>

or

<pre>
$ conda install -r requirements.txt
</pre>



## Demo
You will find a 'test_simple.sh' script at the root (./test_simple.sh).
As your mouse's pointer moves, a safe exit is enabled by forcing your pointer in the upper-left corner.
Move your curser up there and the application will terminate safely.

As shown below, the current example in test_simple.sh will infer from the demo video, and
use FP32 precision for Facial, Head and Gaze models. The Face model's default is FP32-INT1.
The second line running the gui.py file helps to output the ./log/benchmark.log metrics.
You can have a look at ./log/results.txt for the table format and metrics shown. 

While running, there is no visual output. To change that, add the flag --outputShow with 'face' for example.

<pre>
python3 ./src/main_computer_pointer_controller.py -itype 'video' -ipath './bin/demo.mp4' \
                              -facialPrecision 'FP32' -headPrecision 'FP32' -gazePrecision 'FP32'
python3 ./utils/gui.py
</pre>

## Documentation 
#### Diagrams
If you are curious, feel free to check the PDFs for additional overview of the project through diagrams.

#### Command Line Arguments Explained
We can set the type and path to the input, and the device on which the inference is done.

There is a folder './models' at the root to place the models' files. The application expects a specific structure.
If you download the files with the Intel OpenVINO downloader and set the './models' as destination folder,
the downloader will create a subfolder called 'intel' (i.e. that's what the -modelSource flag refers to). In addition,
the downloader will create subsequent subfolder inside 'intel', one for each model based on their name. Finally,
for each model folder, the downloader will create a subfolder per precision and in it, place the two files .bin and .xml.

To keep this clarity, we make use of two flags per model, say -faceModel and -facePrecision, in such a way: 
"./models/intel/<faceModel>/<facePrecision>/<faceModel>.extensions"

where intel refers to our -modelSource flag. Do not specify any extensions.

By default (no argument specified, contrary to the demo above), the app run with the webcam on a CPU with Intel OpenVINO models in FP32: 

<pre>usage: main_computer_pointer_controller.py [-h] [-itype INPUTTYPE]
                                           [-ipath INPUTPATH] [-d DEVICE]
                                           [-modelSource MODELSOURCE]
                                           [-faceModel FACEMODEL]
                                           [-facePrecision FACEPRECISION]
                                           [-facialModel FACIALMODEL]
                                           [-facialPrecision FACIALPRECISION]
                                           [-headModel HEADMODEL]
                                           [-headPrecision HEADPRECISION]
                                           [-gazeModel GAZEMODEL]
                                           [-gazePrecision GAZEPRECISION]
                                           [-g GUI] [-o OUTPUTSHOW]
</pre>

| arguments | explanation |
| --------- | ----------- |
| -itype | The type of input. 'video', 'image' or 'cam for webcam. Default='cam'. |
| -ipath | The path to the input file. Default=0, for ewbcam. |
| -d, --device | The device name: CPU (default), GPU or MYRIAD (for VPU). |
| | |
| -modelSource | Represents a subfolder in the subfolder ./models where you store your models. Default='intel'. |
| | |
| -faceModel | The name of the face detection model in the subfolder ./models/intel/<faceModel>/<facePrecision>/<faceModel>.extensions. Default=face-dtection-adas-binary-0001 |
| -facePrecision | The model precision for face detection model (cf arg faceModel_desc). Default=FP32-INT1 |
| -facialModel | The name of the facial landmarks detection model in the subfolder ./models/intel/<facialModel>/<facialPrecision>/<facialModel_desc>.extensions. Default=landmarks-regression-retail-0009 |
| -facialPrecision | The model precision for facial landmarks detection model. Default=FP32 |
| -headModel | The path to the head pose estimation model in the subfolder ./models/intel/<headModel>/<headPrecision>/<headModel>.extensions. Default=head-pose-estimation-adas-0001 |
| -headPrecision | The model precision for head pose estimation model. Default=FP32 |
| -gazeModel | The path to the gaze estimation model in the subfolder ./models/intel/<gazeModel>/<gazePrecision>/<gazeModel>.extensions. Default=gaze-estimation-adas-0002 |
| -gazePrecision | The model precision for gaze estimation model. Default=FP32 |
| | |
| -o, --outputShow | a string to show single output for 'face, 'facial', 'head', or 'gaze'. Also possible 'faceFacial'. Default='' for no output. |
| -g GUI, --gui GUI | Set to True to activate the graphic user interface (GUI). Default=False. |


## Benchmarks
Results.txt in the ./log folder contains several metrics and test conducted.
In terms of hardware, the application has only been tested on a CPU. The next step will be to test it on a VPU 
(such as Intel Neural Compute Stick 2).

Except for face's model which come in one precision only, all other models show a relative drop in inference time
when I change the precision from FP32 to FP16. The drop in accuracy hasn't been recorded.
Interested reader can refer to OpenVINO's performance information at this [link](docs.openvinotoolkit.org/latest/_docs_performance_int8_vs_fp32.html)

One example from results.txt with Face in FP32-INT1, FP32-INT8 for all:

| (in sec) | model loading time | input processing time | model inference time | output processing time |
| ---------- | ------------------ | --------------------- | -------------------- | ---------------------- |
| Face | 0.2296 | 0.2296 | 0.0249 | 0.0016 |
| Facial Landkmarks | 0.0809 | 0.0001 | 0.0007 | 0.0017 |
| Head Pose | 0.0941 | 0.0001 | 0.0023 | 0.0000| 
| Gaze | 0.1128 | 0.0000 | 0.0027 | 0.0000 |


## Results
Let's start with the precision level (FP32, FP16, INT8) of our models. They refer to the number format such as single-precision (32-bit) 
floating point format for the 'FP32' precision, and half-precision (16-bit) floating point format for the 'FP16' precision. 
They actually represent the weights or parameters that our pre-trained models take to infer on our input video.

In theory, having more precision, the FP32 should come with higher accuracy in the inference tasks. Whereas the FP16
provides a precision that is at a lower level, we can expect models with FP16 precision to require less memory and see 
a drop in inference time. In practice, the differences may be more difficult to analyze due to hardware (CPU, GPU,...)
own limitations, or model's specificity and complexity for example.

## Per Layer Performance
One detailed analysis per model is logged in the file ./log/layers.log.
This is where you can find performances details about each layers of each model.
(the limit of one analysis per model is due to file's size becoming too large by the end of the inference)

### Edge Cases
The application is still sensitive to some situation such as multiple people in frame.
While using the webcam, sudden move of the subject is to be discouraged as the inference is not possible anymore.
A strong (counter)exposition to the sun or a light will also impact negatively the process.


### Models: base models specification, input/output formats
A brief summary of the official links (see model's section above).

| model | GFlops |  MParams | Source framework | Input | Output Inference Engine format |
| ----- | ------ |  ------- | ---------------- | ----- | ------ |
| face-detection-adas-binary-0001 |  0.611 | 1.053 | PyTorch | [1 batch x 3 channels x 384 H x 672 W], BGR | [1, 1, N bboxes, 7] |
| head-pose-estimation-adas-0001 |   0.105 |  1.911 | Caffe | [1x3x60x60], BGR | angle_y_fc=[1,1]; angle_p_fc=[1,1]; angle_r_fc=[1,1] |
| landmarks-regression-retail-0009 |  0.021 |  0.191 | PyTorch | [1x3x48x48], BGR | blob [1, 10] |
| gaze-estimation-adas-0002 |  0.139 |  1.882 | Caffe2 | left_eye_image[1x3x60x60], right_eye_image[1x3x60x60], head_pose_angles[1x3] | gaze_vector[1x3] |

####head-pose:
"Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll)."

####landmarks-regression-retail: 
"mean normed error (on VGGFace2) 0.0705)", "Face location requirements: tight crop",
"the net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
coordinates in the form (x0,y0, x1,y1, ..., x5,y5). All the coordinates are normalized to be in range [0,1]."

####gaze-estimation:
"The net outputs a blob with the shape:[1,3], containing Cartesian coordinates of gaze direction vector.
Please note that the output vector is not normalized and has non-unit length."

## Possible improvements 
The application can be improved through:
* refactoring the code such as the models classes
* create more robust inferences: signal to the user if there are more than one person and which one is influencing the curser
* creating a user-interface for ease in testing different options
* analyzing in details the accuracy of the models
* ...
