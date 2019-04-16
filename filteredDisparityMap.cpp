#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp" 

#include <stdio.h>
#include <iostream>
#include <string.h>

//to compile: g++ filteredDisparityMap.cpp -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_ximgproc -o veRun

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance){
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

int main(int, char**){

	bool no_display;
    bool no_downscale;
    int max_disp, wsize; //Stereo correspondence parameters 
    double lambda, sigma; //Post-filter parameters
    double vis_mult; //Coefficient used for Disparity Map (DM) visualization scale
    Mat imgDisparity8U;
    double minVal, maxVal;
    String filter;
    String algo; //Which OpenCV algorithm was used, BM or SGBM
    String dst_path; //Optional path to save filtered DM result
    String dst_raw_path; //Optional to save DM without filter
	String dst_conf_path; //Optional path to save the trust map used for filtering
	
	char key = 0;

	Ptr<DisparityWLSFilter> wls_filter;
   	double matching_time, filtering_time;

	Mat m_imageRight, m_imageLeft, img1, img2;
	Mat imgU1, imgU2, grayDisp1, grayDisp2;

	Mat GT_disp, left_for_matcher, right_for_matcher;
   	Mat left_disp, right_disp, filtered_disp, conf_map;
    
   	Mat filtered_disp_vis, raw_disp_vis;
   	Mat imgCalorHSV, imgAdd, imgCalorHOT, imgCalorBONE;

	VideoCapture videoOne("/City/City/2011_09_29_2/image_00/data/%10d.png"); //Absolute path to the KITTI left grey frames

	VideoCapture videoTwo("/City/City/2011_09_29_2/image_01/data/%10d.png"); //Absolute path to the KITTI right grey frames

	int width, height;

	width = videoOne.get(3);
	height = videoOne.get(4);

	cout << "width: " << width << endl;
	cout << "height: " << height << endl;


	VideoWriter videoOutAllTwo, videoOutAllFour, videoOutAllFive, videoOutAllSix;

	videoOutAllTwo = cv::VideoWriter("originalEsq.avi",CV_FOURCC('M','J','P','G'), 30, Size(width,height),true); //To save the original left images as video
	videoOutAllFive = cv::VideoWriter("MDBONE.avi",CV_FOURCC('M','J','P','G'), 30, Size(width,height),true); //To save the DM results as BONE colormap video.
	videoOutAllSix = cv::VideoWriter("MDHOT.avi",CV_FOURCC('M','J','P','G'), 30, Size(width,height),true); //To save the DM results as HOT colormap video.

while(1){

  	videoOne >> m_imageLeft;
  	videoTwo >> m_imageRight;
	
	//Histogram equalization to deal with illumination problems
	/*cv::cvtColor(m_imageLeft, m_imageLeft, CV_BGR2Lab);
	std::vector<cv::Mat> channels;
	cv::split(m_imageLeft, channels);
	cv::equalizeHist(channels[0], channels[0]);
	cv::merge(channels, m_imageLeft);
	cv::cvtColor(m_imageLeft, m_imageLeft, CV_Lab2BGR);


	cv::cvtColor(m_imageRight, m_imageRight, CV_BGR2Lab);
	std::vector<cv::Mat> channelsTwo;
	cv::split(m_imageRight, channelsTwo);
	cv::equalizeHist(channels[0], channelsTwo[0]);
	cv::merge(channelsTwo, m_imageRight);
	cv::cvtColor(m_imageRight, m_imageRight, CV_Lab2BGR);*/
	//Histogram equalization ends here

 	if(!m_imageLeft.data || !m_imageRight.data)
 	{
   		printf( " No image data \n " );
   		return -1;
 	}
	
	imgDisparity8U = Mat(m_imageRight.rows, m_imageRight.cols, CV_8UC1);
    filter = "wls_conf"; //Post-filter
    algo = "sgbm"; //Defines which OpenCV algorithm was used, BM or SGBM
    dst_path = "None";
    dst_raw_path = "None";
    dst_conf_path = "None";

	max_disp = 160; //160
    lambda = 8000.0;
    sigma = 3.5;
    vis_mult = 3.0;
    
    wsize = 3; // 3 if SGBM
    //wsize = 15; // if BM, 7 or 15

	conf_map = Mat(m_imageLeft.rows,m_imageLeft.cols,CV_8U);
	conf_map = Scalar(255);
    Rect ROI;

	//Results better than "wls_no_conf"
    if(filter=="wls_conf"){
        if(!no_downscale){ //This is done to leave faster, but for a better result, avoid using.
            max_disp/=2;
            if(max_disp%16!=0){
                max_disp += 16-(max_disp%16);
            }
            resize(m_imageLeft, left_for_matcher, Size(), 0.5, 0.5);
            resize(m_imageRight, right_for_matcher, Size(), 0.5, 0.5);
        }else{
            left_for_matcher = m_imageLeft.clone();
            right_for_matcher = m_imageRight.clone();
        }
        
        /* The filter instance is created by providing the instance of the StereoMatcher
         * Another instance is returned by createRightMatcher. These two instances are 
         * used to calculate the DM's for the right and left images, this is necessary 
         * for filtering afterwards.
         */
        if(algo=="bm"){
            Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
            
            cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
            
            matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }else if(algo=="sgbm"){
            Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
            left_matcher->setP1(24*wsize*wsize);
            left_matcher->setP2(96*wsize*wsize);
            left_matcher->setPreFilterCap(63);
            left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

            matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        
        /* Filter
         * MD calculated by the respective match instances, just as the 
         * left image is passed to the filter. 
         * Note that we are using the original image to guide the filtering 
         * process.
         */
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp, m_imageLeft, filtered_disp, right_disp);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
        
        conf_map = wls_filter->getConfidenceMap();
        
        //Get the ROI that was used in the last filter call:
        ROI = wls_filter->getROI();
        if(!no_downscale)
        {
            //Upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp,left_disp,Size(),2.0,2.0);
            left_disp = left_disp*2.0;
            ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
        }
    }
    
    else if(filter=="wls_no_conf"){
        /* There is no convenience function for the case of filtering with no confidence, so we
        will need to set the ROI and matcher parameters manually */

        left_for_matcher  = m_imageLeft.clone();
        right_for_matcher = m_imageRight.clone();

        if(algo=="bm"){
            Ptr<StereoBM> matcher  = StereoBM::create(max_disp,wsize);
            matcher->setTextureThreshold(0);
            matcher->setUniquenessRatio(0);
            cvtColor(left_for_matcher,  left_for_matcher, COLOR_BGR2GRAY);
            cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        else if(algo=="sgbm")
        {
            Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
            matcher->setUniquenessRatio(0);
            matcher->setDisp12MaxDiff(1000000);
            matcher->setSpeckleWindowSize(0);
            matcher->setP1(24*wsize*wsize);
            matcher->setP2(96*wsize*wsize);
            matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
            ROI = computeROI(left_for_matcher.size(),matcher);
            wls_filter = createDisparityWLSFilterGeneric(false);
            wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

            matching_time = (double)getTickCount();
            matcher->compute(left_for_matcher,right_for_matcher,left_disp);
            matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,m_imageLeft,filtered_disp,Mat(),ROI);
        filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    }
    
    //collect and print all the stats:
    //cout.precision(2);
    //cout<<"Matching time:  "<<matching_time<<"s"<<endl;
    //cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
    //cout<<endl;

    if(dst_path!="None"){
        //Mat filtered_disp_vis;
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        imwrite(dst_path,filtered_disp_vis);
    }
    if(dst_raw_path!="None"){
        //Mat raw_disp_vis;
        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        imwrite(dst_raw_path,raw_disp_vis);
    }
    if(dst_conf_path!="None"){
        imwrite(dst_conf_path,conf_map);
    }
    
    if(!no_display)
    {
        /*//Displays the original images
        namedWindow("left", WINDOW_AUTOSIZE);
        imshow("left", imgLeft);
        namedWindow("right", WINDOW_AUTOSIZE);
        imshow("right", imgRight);*/

        /*if(!noGT)
        {
            Mat GT_disp_vis;
            getDisparityVis(GT_disp,GT_disp_vis,vis_mult);
            namedWindow("ground-truth disparity", WINDOW_AUTOSIZE);
            imshow("ground-truth disparity", GT_disp_vis);
        }*/

        /*//Displays DM without filter
        Mat raw_disp_vis;
        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        namedWindow("raw disparity", WINDOW_AUTOSIZE);
        imshow("raw disparity", raw_disp_vis);*/
        
        //Displays filtered DM
        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        namedWindow("filtered disparity", WINDOW_AUTOSIZE);
        imshow("filtered disparity", filtered_disp_vis);
        
         /* Color Maps:
          * OpenCV method to change a grayscale image to a color model.
          * The human vision may have difficulty perceiving small 
          * differences in shades of gray, but better perceives the 
          * changes between colors.
          * 
          * More Info: http://docs.opencv.org/3.1.0/d3/d50/group__imgproc__colormap.html#gsc.tab=0
          */
        //Applying color maps (different DM visualization)
		applyColorMap(filtered_disp_vis, imgCalorBONE, COLORMAP_BONE);
		applyColorMap(filtered_disp_vis, imgCalorHOT, COLORMAP_HOT);

		//imshow("Left image", m_imageLeft);
		//imshow("Right image", m_imageRight);
		
		videoOutAllTwo.write(m_imageLeft);
		videoOutAllFive.write(imgCalorBONE);
		videoOutAllSix.write(imgCalorHOT);

		key = (char) waitKey(5);
		if(key==27){
			break;
		}
	}
	}
	return 0;
}
