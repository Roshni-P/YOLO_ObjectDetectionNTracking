# YOLO_ObjectDetectionNTracking
Given a video, different objects are detected and tracked with a unique ID using YOLOv8.<br>
The model inference, processing, rendering is done in C++. The C++ code is exposed as a Shared Object file in Python.<br>

<img width="800" height="453" alt="Screencastfrom06-10-26032954PMIST-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/acbcaeb6-9ee0-4f8c-8714-40016e8998b7" />


**Steps To Build**<br>
1. Remove cached object files and build directories <br>
**rm -rf build/ dist/ *.so src/*.o**<br>

2. Rebuild in-place<br>
**python setup.py build_ext --inplace** <br>

3. Run the main python file<br>
**python mainWithCpp.py** <br>

**Note:** <br>
The project has a different set of files. One main.py, that captures the video frames in python and pass each frame to yoloDetector.cpp. Inferencing is done in C++ while capturing and rendering occurs in Python. The drawback of this approach was that there is a lag due to passing of frames between Python and C++. This lag has been eliminated in mainWithCpp.py and yoloWithCpp.cpp
