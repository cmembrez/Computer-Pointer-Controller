# Computer Pointer Controller

The current application reads an input (webcam, video) and will impact the user mouse controller with respect to the user's gaze.
To estimate the user's gaze, four pre-trained deep learning models are used.

## Project Set Up and Installation
#### Explaining the setup procedures to run the project
First start by cloning the current respository:
git clone www.github.com/cmembrez/computer_pointer_controller

Run it in a virtual environment (...).

Let your venv install any missing packages listed in the ./requirements.txt (see directory structure below).

Place four models in the ./models folder. You can download pre-trained models from Intel OpenVINO resources.

In a terminal, launch the *main_computer_pointer_controller.py*.

Running it without arguments mean that you have the four following models and will run through your webcam:

    i_desc = "The location of the input file. Default=0, for webcam."
    d_desc = "The device name: CPU (default), GPU, MYRIAD (for VPU)."

    modelSource_desc = "Represents a subfolder in the subfolder ./models where you store your models. Default='intel'."

    faceModel_desc = "The name of the face detection model in the subfolder ./models/intel/<faceModel>/<facePrecision>/<faceModel>.extensions. Default=face-dtection-adas-binary-0001"
    facePrecision_desc = "The model precision for face detection model (cf arg faceModel_desc). Default=FP32-INT1"

    facialModel_desc = "The name of the facial landmarks detection model in the subfolder ./models/intel/<facialModel>/<facialPrecision>/<facialModel_desc>.extensions. Default=landmarks-regression-retail-0009"
    facialPrecision_desc = "The model precision for facial landmarks detection model. Default=FP32"

    headModel_desc = "The path to the head pose estimation model in the subfolder ./models/intel/<headModel>/<headPrecision>/<headModel>.extensions. Default=head-pose-estimation-adas-0001"
    headPrecision_desc = "The model precision for head pose estimation model. Default=FP32"

    gazeModel_desc = "The path to the gaze estimation model in the subfolder ./models/intel/<gazeModel>/<gazePrecision>/<gazeModel>.extensions. Default=gaze-estimation-adas-0002"
    gazePrecision_desc = "The model precision for gaze estimation model. Default=FP32"
    
Please consult the log file ('./log/main.log') for details on any error happening while running the application.

#### project directory structure

<pre>
├── bin
│   └── demo.mp4
├── log
│   ├── main.log
│   └── benchmark_time.log
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
    ├── rotation3d.py
    └── tools_image.py
</pre>

#### models you need to download 

#### where to place them etc. 

If you download four models face-detection-adas-binary-0001, gaze-estimation-adas-0002,
head-pose-estimation-adas-0001, landmarks-regression-retail-0009, with the Intel OpenVINO downloader into ./models,
you would have the following structure in the models folder:

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
        ├── head-pose-estimation-adas-0001
        │   ├── FP16
        │   │   ├── head-pose-estimation-adas-0001.bin
        │   │   └── head-pose-estimation-adas-0001.xml
        │   ├── FP32
        │   │   ├── head-pose-estimation-adas-0001.bin
        │   │   └── head-pose-estimation-adas-0001.xml
        │   └── FP32-INT8
        │       ├── head-pose-estimation-adas-0001.bin
        │       └── head-pose-estimation-adas-0001.xml
        └── landmarks-regression-retail-0009
            ├── FP16
            │   ├── landmarks-regression-retail-0009.bin
            │   └── landmarks-regression-retail-0009.xml
            ├── FP32
            │   ├── landmarks-regression-retail-0009.bin
            │   └── landmarks-regression-retail-0009.xml
            └── FP32-INT8
                ├── landmarks-regression-retail-0009.bin
                └── landmarks-regression-retail-0009.xml
</pre>

#### how to install the dependencies your project requires.


## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. 
#### Explaining the command line arguments that the project supports
We can set the input (either webcam or video file) and the device on which the inference is done.
Then, we can set a subfolder name that is inside the .models/ folder. If you use Intel downloader, it will create one.
Finally, for each of the four models, we provide one model name argument (exclude any .bin, .xml extensions),
and the precision used (e.g. FP32-INT1, FP32, FP16, ...).
By default (no argument specified), the app run with the webcam on a CPU with Intel OpenVINO models in FP32: 

| arguments | explanation |
| --------- | ----------- |
| i_desc | The location of the input file. Default=0, for webcam. |
| d_desc | The device name: CPU (default), GPU, MYRIAD (for VPU). |
| | |
| modelSource_desc | Represents a subfolder in the subfolder ./models where you store your models. Default='intel'. |
| | |
| faceModel_desc | The name of the face detection model in the subfolder ./models/intel/<faceModel>/<facePrecision>/<faceModel>.extensions. Default=face-dtection-adas-binary-0001 |
| facePrecision_desc | The model precision for face detection model (cf arg faceModel_desc). Default=FP32-INT1 |
| facialModel_desc | The name of the facial landmarks detection model in the subfolder ./models/intel/<facialModel>/<facialPrecision>/<facialModel_desc>.extensions. Default=landmarks-regression-retail-0009 |
| facialPrecision_desc | The model precision for facial landmarks detection model. Default=FP32 |
| headModel_desc | The path to the head pose estimation model in the subfolder ./models/intel/<headModel>/<headPrecision>/<headModel>.extensions. Default=head-pose-estimation-adas-0001 |
| headPrecision_desc | The model precision for head pose estimation model. Default=FP32 |
| gazeModel_desc | The path to the gaze estimation model in the subfolder ./models/intel/<gazeModel>/<gazePrecision>/<gazeModel>.extensions. Default=gaze-estimation-adas-0002 |
| gazePrecision_desc | The model precision for gaze estimation model. Default=FP32 |
| --------- | ------------ |

## Benchmarks
#### Include the benchmark results of running your model on multiple hardwares 

#### and multiple model precisions. 

#### Your benchmarks can include: 
### local no visual outputs, CPU, webcam, FP32-INT1 for Face and FP32 for other three (not the FP32-INT8)
| model name | model loading time | input processing time | model inference time | output processing time |
| ---------- | ------------------ | --------------------- | -------------------- | ---------------------- |
| Face | 0.2072 | 0.0014 | 0.0338 | 0.0015 |
| Facial Landkmarks | 0.0749 | 0.0002 | 0.0008 | 0.0003 |
| Head Pose | 0.0882 | 0.0000 | 0.0059 | 0.0000| 
| Gaze | 0.1129 | 0.0000 | 0.0096 | 0.0000 |


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

| cause | description | solution |
| ----- | ----------- | -------- |
| blur | E.g., when the webcam needs to focus again, the frame is blur (between two 'focuses and sharp' frame) and model detections cannot detect anymore. | can the subject calm down :)? ; get auto-focus webcam; clean webcam lens; |
 
 ? what about glasses? medical/sun/... covid mask?

### Models: base models specification, input/output formats

| model | min head size | GFlops | GI1ops | MParams | Source framework | Input | Output Inference Engine format |
| ----- | ------------- | ------ | ------ | ------- | ---------------- | ----- | ------ |
| face-detection-adas-binary-0001 | 90x90 on 1080p | 0.611 | 2.224 | 1.053 | PyTorch | [1 batch x 3 channels x 384 H x 672 W], BGR | [1, 1, N bboxes, 7] |
| head-pose-estimation-adas-0001 | na | 0.105 | na | 1.911 | Caffe | [1x3x60x60], BGR | angle_y_fc=[1,1]; angle_p_fc=[1,1]; angle_r_fc=[1,1] |
| landmarks-regression-retail-0009 | na | 0.021 | na | 0.191 | PyTorch | [1x3x48x48], BGR | blob [1, 10] |
| gaze-estimation-adas-0002 | na | 0.139 | na | 1.882 | Caffe2 | left_eye_image[1x3x60x60], right_eye_image[1x3x60x60], head_pose_angles[1x3] | gaze_vector[1x3] |

####head-pose:
"each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll)."

####landmarks-regression-retail: 
"mean normed error (on VGGFace2) 0.0705)", "Face location requirements: tight crop",
"the net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
coordinates in the form (x0,y0, x1,y1, ..., x5,y5). All the coordinates are normalized to be in range [0,1]."

####gaze-estimation:
"The net outputs a blob with the shape:[1,3], containing Cartesian coordinates of gaze direction vector.
Please note that the output vector is not normalized and has non-unit length."
