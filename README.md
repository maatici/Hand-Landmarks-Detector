# Hand Landmarks Detector for Hand Analysis

Hand research is one of the hot topics and currently there exists a number of works in literature that investigate the correlation between digit ratio (2D:4D) and some other traits of a person. Main objective of this project is detecting the hand and calculating the ratios used for hand analysis automatically. For this, I have designed models that detect the hand box  and 9 landmarks of the hand in jpg/png images.

### Installing

This project is developed on Ubuntu 16.04 by using: 

1. [OpenCV 3.3](https://opencv.org)
2. [Dlib 19.7](http://dlib.net)

You can follow the instructions [here](http://www.learnopencv.com/install-opencv3-on-ubuntu/) to install OpenCV. After successful installation of OpenCV;

Download Dlib from [here](http://dlib.net) 
Download this repository
Edit CmakeLists.txt according to your Dlib directory

Open a terminal and change the directory to this repository and run following commands:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/handDetectionExample path/to/hand/images
./build/handLandmarkDetectionExample path/to/hand/images
```

## License

The handTrainer.cpp and handLandmarkTrainer.cpp files are allowed to use in any way that [Dlib License](http://dlib.net/license.html) permits. Please note that these trainer files just show my optimized parameters and are only useful if you prepare your own dataset. The files apart from the handTrainer.cpp and handLandmarkTrainer.cpp are only allowed for academic use. I trained the HandDetector.svm and Hand_9_Landmarks_Detector.dat models by using the images from datasets [11K Hands](https://sites.google.com/view/11khands), [The NUS hand posture datasets I & II](https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/) and [MOHI](https://www.mutah.edu.jo/biometrix/hand-images-databases.html). Therefore usage of models are also subject to permissions of owners of these datasets.

