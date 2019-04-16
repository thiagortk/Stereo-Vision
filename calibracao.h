#ifndef CALIBRACAOCAMERA_H
#define CALIBRACAOCAMERA_H

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

class calibracao {

public:
    calibracao(cv::Mat& actualOne, cv::Mat& actualTwo);
    ~calibracao();
    
    void iniciaCalibracaoCamera();
    
private:

    //Chessboard Settings
    int numBoards = 13; //Number of images for the calibration
    int board_w = 9; //Horizontal corners
    int board_h = 6; //Vertical corners
    float squareSize = 2.5f; //Standard = 1. Small chessboard = 2,5. Large chessboard = 4,4
    
    vector<vector<Point3f>> object_points; //Represents the 3D corners actual location
    vector<vector<Point2f>> imagePoints1, imagePoints2; //Represent the location of corners detected in 3D
    vector<Point2f> corners1, corners2;
    vector<Point3f> obj;
    
    Mat img1, img2, gray1, gray2;
    Mat m_imageOne, m_imageTwo;
    
    // CM's are 3x3 floating point arrays of each camera
    // D's are distortion coefficients vectors of each camera
    // D's Matrix of distortion coefficient of camera 1 and 2
    Mat CM1, CM2, D1, D2;
    
    // R - Rotation Matrix between the first and second camera coordinate systems
    // T - Translation vector between the cameras coordinate systems
    // E - Essential Matrix
    // F - Fundamental matrix
    Mat R, T, E, F;
    
    // R1 - 3x3 Rectification Transformation (Rotation Matrix) for the first Camera
    // R2 - 3x3 Rectification Transformation (Rotation Matrix) for the second Camera
    // P1 - Projection matrix 3x4 in the new and rectified coordinate system of the first camera
    // P2 - Projection matrix 3x4 in the new and rectified coordinate system of the second camera
    // Q - Disparity matrix by depth 4x4
    Mat R1, R2, P1, P2, Q; 
    
};

#endif /* CALIBRACAOCAMERA_H */
