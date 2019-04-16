#ifndef DISPARIDADE_H
#define DISPARIDADE_H

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp" 
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <stdio.h>
#include <iostream>
#include <string.h>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace cv::xfeatures2d;

class disparidade {
public:
    disparidade(Mat& actualOne, Mat& actualTwo);
    ~disparidade();
    
    void iniciaDisparidade(); //Loads calibration and rectification data

private: 
    void retificaParaDisparidade(Mat map1x, Mat map1y, Mat map2x, Mat map2y); //Uses saved calibration and rectification data to rectify the images to the Disparity Map.
    void constroiMapaDisparidadeBM(Mat imgRight, Mat imgLeft); //Basic, using OpenCV BM and without filters
    void constroiMapaDisparidadeSGBM(Mat imgRight, Mat imgLeft); //Basic, using OpenCV SGBM and without filters
    void constroiMDFiltro(Mat imgRight, Mat imgLeft); //Example with filter available in extra modules of OpenCV 3.1 or later
    Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance); //Region of interest, used for the above example, OpenCV with filter
    
    bool no_display;
    bool no_downscale;
    int max_disp, wsize; //Stereo matching parameters
    double lambda, sigma; //Post-filter parameters
    double vis_mult; //Coefficient used for DM visualization scale
    Mat imgDisparity8U;
    double minVal, maxVal;
    String filter;
    String algo;
    String dst_path;
    String dst_raw_path;
    String dst_conf_path;
    
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;
    
    Mat m_imageRight, m_imageLeft, img1, img2, imgOutOne, imgOutTwo;
    Mat imgU1, imgU2, grayDisp1, grayDisp2;
    
    Mat GT_disp, left_for_matcher, right_for_matcher;
    Mat left_disp, right_disp, filtered_disp, conf_map;
    
    cv::Mat filtered_disp_vis, raw_disp_vis;
    Mat imgCalorHSV, imgCalorJET, imgAdd, imgBONE, imgHOT;
    
    cv::VideoWriter videoOutDispatiryJET;
    cv::VideoWriter videoOutDisparityHSV;
    cv::VideoWriter videoOutDisparityBONE;
    cv::VideoWriter videoOutDisparityHOT;
    cv::VideoWriter videoOutOriginalOne;
    cv::VideoWriter videoOutOriginalTwo;
    
};

#endif /* DISPARIDADE_H */

