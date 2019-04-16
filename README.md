# stereoVision
C++ example codes for camera calibration, rectification and to build disparity maps

Must have OpenCV 3.1 or later installed with extra modules.

The following examples were used:

- Stereo BM method: http://docs.opencv.org/3.1.0/d9/dba/classcv_1_1StereoBM.html#details&gsc.tab=0
- Stereo SGBM method: http://docs.opencv.org/3.1.0/d2/d85/classcv_1_1StereoSGBM.html#details
- Disparity map post-filtering (extra modules): https://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html#gsc.tab=0
- Color maps: http://docs.opencv.org/3.1.0/d3/d50/group__imgproc__colormap.html#gsc.tab=0

If you are going to use real-time images and perform your own calibration, use the class: "*calibracao.cpp*" and then the "*disparity.cpp*". Build your main

If you are going to use already rectified data set images, you can use "*filteredDisparityMap.cpp*". The main is inside.
**Caution**: it has absolute path in the code for the rectified images that should be used for the construction of the Disparity Map. You will need to change the absolute paths or change to use argv.

**To compile**: g++ filteredDisparityMap.cpp -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_ximgproc -o veRun
