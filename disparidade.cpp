#include <opencv2/video/tracking.hpp>

#include "disparidade.h"

disparidade::disparidade(Mat& actualOne, Mat& actualTwo) : m_imageRight(actualOne), m_imageLeft(actualTwo) {
    /*To save the Disparity Map in different views.*/
    //videoOutDispatiryJET = cv::VideoWriter("/media/thiago/Lobinho/outDispatiryJET.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);
    videoOutDisparityHSV = cv::VideoWriter("outDispatiryHSV.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);
    videoOutDisparityBONE = cv::VideoWriter("outDispatiryHSV.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);
    videoOutDisparityHOT = cv::VideoWriter("outDispatiryHSV.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);
    videoOutOriginalOne = cv::VideoWriter("outTestDMDOne.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);
    videoOutOriginalTwo = cv::VideoWriter("outTestDMDTwo.avi",CV_FOURCC('M','J','P','G'), 10, Size(352,288),true);

    if (!videoOutDisparityBONE.isOpened())
    {
        std::cout << "Could not save the video." << std::endl;
    }
    
    if (!videoOutDisparityHOT.isOpened())
    {
        std::cout << "Could not save the video." << std::endl;
    }
    
    if (!videoOutOriginalOne.isOpened())
    {
        std::cout << "Could not save the video." << std::endl;
    }
    
    if (!videoOutOriginalTwo.isOpened())
    {
        std::cout << "Could not save the video." << std::endl;
    }
}

disparidade::~disparidade() {
    m_imageRight.release();
    m_imageLeft.release();
    img1.release();
    img2.release();
    imgU1.release();
    imgU2.release();
    grayDisp1.release();
    grayDisp2.release(); 
    GT_disp.release();
    left_for_matcher.release();
    right_for_matcher.release();
    left_disp.release();
    right_disp.release();
    filtered_disp.release();
    conf_map.release();
    imgCalorHSV.release();
    imgCalorJET.release();
    imgAdd.release();
    filtered_disp_vis.release();
    raw_disp_vis.release();
}

/* Loads the calibration data that is saved to a file with the
 * data stored from the last calibration. It also loads the 
 * rectification data.
 */
void disparidade::iniciaDisparidade(){
    
    //Mat grayDisp1, grayDisp2; 
    img1 = m_imageRight.clone();
    img2 = m_imageLeft.clone();
    
    cvtColor(img1, img1, CV_BGR2GRAY);
    cvtColor(img2, img2, CV_BGR2GRAY);
    
    //+++++Load calibration data from calibrated file +++++//
    Mat CM1 = Mat(3, 3, CV_64FC1); //Calibration
    Mat CM2 = Mat(3, 3, CV_64FC1); //Calibration
    Mat D1, D2; //Calibration
    Mat R, T, E, F; //Calibration
    
    Mat R1, R2, P1, P2, Q; //Rectification
    
    FileStorage fs("mystereocalib.yml", FileStorage::READ);
    fs["CM1"] >> CM1;
    fs["CM2"] >> CM2;
    fs["D1"] >> D1;
    fs["D2"] >> D2;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["Q"] >> Q;
    cout << Q << endl;    
    
    fs.release();
    
    cout << "Applying Undistort..." << endl;
    
    Mat map1x, map1y, map2x, map2y;
    
    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

    cout << "Undistort completed" << endl;
    
    //It rectifies the images by passing the parameters with the necessary information.
    retificaParaDisparidade(map1x, map1y, map2x, map2y);
        
}

/* Receives the necessary calibration and rectification information and finalizes
 * the images rectification that will be used for the Disparity Map construction.
 */
void disparidade::retificaParaDisparidade(Mat map1x, Mat map1y, Mat map2x, Mat map2y){
    
    Mat imgRectify, imgOne, imgTwo;
    int k;
    char key;
        
    while(1){
        imgOne = m_imageRight.clone();
        imgTwo = m_imageLeft.clone();
        
        cvtColor(imgOne, imgOne, COLOR_BGR2RGB);
        cvtColor(imgTwo, imgTwo, COLOR_BGR2RGB);
        
        remap(imgOne, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(imgTwo, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
          
        /* Rectification View:
         * If you want to check if the loaded rectification is correct.
         *
        // To display the rectified images
        imgRectify = Mat::zeros(imgU1.rows, imgU1.cols*2+10, imgU1.type());
        
        imgU1.copyTo(imgRectify(Range::all(), Range(0, imgU2.cols)));
        imgU2.copyTo(imgRectify(Range::all(), Range(imgU2.cols+10, imgU2.cols*2+10)));
        
        //If it gets too large to fit on the screen, it is scaled down, by 2, to fit.
        if(imgRectify.cols > 1920){
            resize(imgRectify, imgRectify, Size(imgRectify.cols/2, imgRectify.rows/2));
        } 
        //To draw the lines in the rectified image
        for(int j = 0; j < imgRectify.rows; j += 16){
            Point p1 = Point(0,j);
            Point p2 = Point(imgRectify.cols*2,j);
            line(imgRectify, p1, p2, CV_RGB(255,0,0));
        }
        
        imshow("Rectified image", imgRectify);*/
        
        //These are grayscale images that will be used in the DM.
        Mat grayDisp1, grayDisp2;        
        
        cvtColor(imgU1, grayDisp1, CV_RGB2GRAY);
        cvtColor(imgU2, grayDisp2, CV_RGB2GRAY);
        
        imwrite("left.ppm", grayDisp1);
        imwrite("right.ppm", grayDisp2);
        
        //constroiMapaDisparidadeBM(imgU1, imgU2);
        //constroiMapaDisparidadeSGBM(imgU1, imgU2);
        constroiMDFiltro(grayDisp1, grayDisp2); //with grayscale images
        //constroiMDFiltro(imgU1, imgU2); //with rgb images
        
        k = waitKey(5);
        key = (char) waitKey(5);
        
        if(key==27){
            break;
        }
    }
}

/* Here only the StereoBM, the OpenCV disparity method, is used.
 * The StereoBM calculates the disparities using a matching algorithm between blocks.
 * As parameters to the method are passed "numDisparities" which is the search range 
 * of disparities, for each pixel the algorithm must find the best disparity (from zero 
 * (minimum standard) to "numDisparities").
 * The other parameter is "blockSize", which is linear size of the blocks compared by 
 * the algorithm, the size must be odd (as the block is centered on the current pizel). 
 * A larger block size implies a smoother but less accurate DM. 
 * A smaller size results in a more detailed DM, but increases the chance of the algorithm
 * finding erroneous matches between pixels.
 * 
 * Example from: https://github.com/Itseez/opencv/blob/master/samples/cpp/tutorial_code/calib3d/stereoBM/SBM_Sample.cpp
 * More info: http://docs.opencv.org/3.1.0/d9/dba/classcv_1_1StereoBM.html#details&gsc.tab=0
 */
void disparidade::constroiMapaDisparidadeBM(Mat imgRight, Mat imgLeft){  
    
    //while(1){
        Mat grayDisp1, grayDisp2;        
        
        cvtColor(imgRight, grayDisp1, CV_RGB2GRAY);
        cvtColor(imgLeft, grayDisp2, CV_RGB2GRAY);
        
        //imshow("image1", imgU1);
        //imshow("image2", imgU2);
        
        Mat imgDisparity16S = Mat(imgRight.rows, imgRight.cols, CV_16S);
        Mat imgDisparity8U = Mat(imgRight.rows, imgRight.cols, CV_8UC1);

        int ndisparities = 16*1;
        int SADWindowSize = 15;
        
        Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

        sbm->compute(grayDisp1, grayDisp2, imgDisparity16S);
        //imwrite( "test.jpg", imgDisparity16S );
        
        double minVal, maxVal;
        
        minMaxLoc( imgDisparity16S, &minVal, &maxVal);
        
        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
        //imgDisparity16S.convertTo(imgDisparity8U, CV_8U);
        
        //namedWindow("windowDisparity", WINDOW_NORMAL);
        imshow("windowDisparity", imgDisparity8U);
        //imshow("16S", imgDisparity16S);
        
        //key = (char) waitKey(5);
        
        //if(key==27){
        //    break;
        //}
    //}
}

 /* Here, only StereoSGBM, the OpenCV disparity method, is used.
  * The StereoSGBM implements the algorithm of H. Hirschmuller (Heiko Hirschmuller, 
  * Stereo processing by semiglobal matching and mutual information, Pattern Analysis 
  * and Machine Intelligence, IEEE Transactions on, 30 (2): 328-341, 2008).
  * By default the algorithm makes a single pass, you can change the way it navigates 
  * in the mode parameter, but this also increases memory consumption.
  * The algorithm makes the correspondence between blocks, not between simple pixels, 
  * but you can leave the block with size equal to 1, leaving as simple pixel.
  * Some pre- and pos-processing are made.
  * The constructor configures all parameters by default.
  * *** Parameters:
  * ** minDisparity: The lowest possible value of disparity. Usually it is zero, 
  * but sometimes rectification algorithms can change the images, so this parameter 
  * needs to be adjusted accordingly.
  * ** numDisparities: Maximum mismatch minus the minimum disparity. Value always 
  * greater than zero. Must be divisible by 16.
  * ** blockSize: Match block size. must be an odd number greater than or equal to 1. 
  * Usually something between 3 and 11.
  * ** P1: The first control parameter of the smoothness of the disparity.
  * ** P2: The second control parameter of the smoothness of the disparity. The larger 
  * the value the smoother the disparity. P1 is the penalty for changing the disparity 
  * by plus or minus 1 between neighboring pixels. P2 is the penalty on the disparity 
  * change in more than one between neighboring pixels. P2 must be greater than P1.
  * ** disp12MaxDiff: Maximum allowable difference (units of integer pixels) in checking 
  * for disparities between left and right images. To disable this check, simply set it 
  * to a non-positive value.
  * ** preFilterCap: Truncation value for the pre-filtered pixels of the image.
  * ** uniquenessRatio: Margin in percentage where the best (minimum) value of the 
  * calculated cost function must "earn" the second best value to consider in the stereo 
  * match found to be correct. Usually a value between 5 and 15 is good enough.
  * ** speckleWindowSize: Maximum size of regions of smooth disparity to consider noise 
  * spots and invalidate. To disregard this filter, set to zero, otherwise leaves a range 
  * between 50 and 200.
  * ** speckleRange: Maximum variation of disparity between related components. If you use 
  * the noise filter set as positive, it should be a multiple of 16. Normally 1 or 2 is 
  * good enough.
  * ** mode: Sets how the algorithm is passed through the image (scanning). Depending on 
  * the option, can greatly increase memory consumption.
  * 
  * More info: http://docs.opencv.org/3.1.0/d2/d85/classcv_1_1StereoSGBM.html#details
  */
void disparidade::constroiMapaDisparidadeSGBM(Mat imgRight, Mat imgLeft){
    
    //while(1){
        Mat grayDisp1, grayDisp2;        
        
        cvtColor(imgRight, grayDisp1, CV_RGB2GRAY);
        cvtColor(imgLeft, grayDisp2, CV_RGB2GRAY);
        
        //imshow("image1", imgU1);
        //imshow("image2", imgU2);
        
        Mat imgDisparity16S = Mat(imgRight.rows, imgRight.cols, CV_16S);
        Mat imgDisparity8U = Mat(imgRight.rows, imgRight.cols, CV_8UC1);

        int ndisparities = 16*1;
        int SADWindowSize = 3;
        
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, ndisparities, SADWindowSize);
        
        sgbm->setP1(24*SADWindowSize*SADWindowSize);
        sgbm->setP2(96*SADWindowSize*SADWindowSize);
        sgbm->setPreFilterCap(63);
        sgbm->setMode(StereoSGBM::MODE_SGBM);
        
        sgbm->compute(grayDisp1, grayDisp2, imgDisparity16S);
        //imwrite( "test.jpg", imgDisparity16S );
        
        double minVal, maxVal;
        
        minMaxLoc( imgDisparity16S, &minVal, &maxVal);
        
        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
        //imgDisparity16S.convertTo(imgDisparity8U, CV_32F, 1.0/16.0, 0.0);
        //imgDisparity16S.convertTo(imgDisparity8U, CV_8U);
        
        //namedWindow("windowDisparity", WINDOW_NORMAL);
        imshow("windowDisparity", imgDisparity8U);
        imshow("16S", imgDisparity16S);
}


/* Method that can use either StereoBM or StereoSGBM, but most important is the
 * post-processing. TO do this a filter is used that smoothes the DM and refines 
 * some occlusions. This filter is available in the extra modules of OpenCV 3.1.
 * 
 * Example from: http://docs.opencv.org/3.1.0/d3/d14/tutorial_ximgproc_disparity_filtering.html#gsc.tab=0
 */
void disparidade::constroiMDFiltro(Mat imgRight, Mat imgLeft){
    
    imgDisparity8U = Mat(imgRight.rows, imgRight.cols, CV_8UC1);
    filter = "wls_conf";
    algo = "sgbm";
    dst_path = "None";
    dst_raw_path = "None";
    dst_conf_path = "None";
    
    max_disp = 5*16;
    lambda = 8000.0;
    sigma = 1.5;
    vis_mult = 1.3;
    
    wsize = 1; // 3 if SGBM
    //wsize = 15; // if BM, 7 or 15
    
    conf_map = Mat(imgLeft.rows,imgLeft.cols,CV_8U);
    conf_map = Scalar(255);
    Rect ROI;
    
    //Better results than "wls_no_conf"
    if(filter=="wls_conf"){
        //if(!no_downscale){ //This is done to make it faster, but for a better result, avoid using it.
        //    max_disp/=2;
        //    if(max_disp%16!=0){
        //        max_disp += 16-(max_disp%16);
        //    }
        //    resize(imgLeft, left_for_matcher, Size(), 0.5, 0.5);
        //    resize(imgRight, right_for_matcher, Size(), 0.5, 0.5);
        //}else{
            left_for_matcher = imgLeft.clone();
            right_for_matcher = imgRight.clone();
        //}
        
        /* The filter instance is created by providing the instance of the 
         * StereoMatcher Another instance is returned by createRightMatcher. 
         * These two instances are used to calculate the DMs for the right 
         * and left images, this is necessary for filtering afterwards.
         */
        if(algo=="bm"){
            Ptr<StereoBM> left_matcher = StereoBM::create(max_disp,wsize);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
            
            //cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
            //cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

            //matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
            //matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }else if(algo=="sgbm"){
            Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
            left_matcher->setP1(8*wsize*wsize);
            left_matcher->setP2(96*wsize*wsize);
            left_matcher->setPreFilterCap(63);
            left_matcher->setMode(StereoSGBM::MODE_HH); //MODE_SGBM_3WAY
            //left_matcher->setBlockSize(1);
            wls_filter = createDisparityWLSFilter(left_matcher);
            Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);
            
            //matching_time = (double)getTickCount();
            left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
            right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
            //matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        }
        
        /* Filter
         * MD calculated by the respective match instances, just as the 
         * left image is passed to the filter. 
         * Note that we are using the original image to guide the filtering 
         * process.
         */
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        //filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp, imgLeft, filtered_disp, right_disp);
        //filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
        
        conf_map = wls_filter->getConfidenceMap();
        
        // Get the ROI that was used in the last filter call:
        ROI = wls_filter->getROI();
        if(!no_downscale)
        {
            // upscale raw disparity and ROI back for a proper comparison:
            resize(left_disp,left_disp,Size(),2.0,2.0);
            left_disp = left_disp*2.0;
            ROI = Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
        }
    }
    
    else if(filter=="wls_no_conf"){
        /* There is no convenience function for the case of filtering with no confidence, so we
        will need to set the ROI and matcher parameters manually */

        left_for_matcher  = imgLeft.clone();
        right_for_matcher = imgRight.clone();

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
        //filtering_time = (double)getTickCount();
        wls_filter->filter(left_disp,imgLeft,filtered_disp,Mat(),ROI);
        //filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    }
    
    //collect and print all the stats:
    /*cout.precision(2);
    cout<<"Matching time:  "<<matching_time<<"s"<<endl;
    cout<<"Filtering time: "<<filtering_time<<"s"<<endl;
    cout<<endl;*/

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
        /*// Displays the original images
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
        //namedWindow("filtered disparity", WINDOW_AUTOSIZE);
        //imshow("filtered disparity", filtered_disp_vis);
        
        /* Color Maps:
         * OpenCV method to change a grayscale image to a color model.
         * The human vision may have difficulty perceiving small 
         * differences in shades of gray, but better perceives the 
         * changes between colors.
         * 
         * More Info: http://docs.opencv.org/3.1.0/d3/d50/group__imgproc__colormap.html#gsc.tab=0
         */
        //Applying color maps (different DM visualization)
        applyColorMap(filtered_disp_vis, imgBONE, COLORMAP_BONE);
        imshow("Color Map BONE", imgBONE);
        applyColorMap(filtered_disp_vis, imgHOT, COLORMAP_HOT);
        imshow("Color Map HOT", imgHOT);
        applyColorMap(filtered_disp_vis, imgCalorHSV, COLORMAP_HSV);
        imshow("Color Map HSV", imgCalorHSV);
        
        cvtColor(m_imageRight, imgOutOne, COLOR_BGR2RGB);
        cvtColor(m_imageLeft, imgOutTwo, COLOR_BGR2RGB);
        videoOutOriginalOne.write(imgOutOne);
        videoOutOriginalTwo.write(imgOutTwo);
        videoOutDisparityBONE.write(imgBONE);
        videoOutDisparityHOT.write(imgHOT);
        videoOutDisparityHSV.write(imgCalorHSV);
        /*//Applying sum with original image and color map (another form of visualization)
        //addWeighted(imgCalorJET, 0.8, imgLeft, 0.2, 1, imgAdd);
        //imshow("Add Weighted", imgAdd);*/
        
        //waitKey();
        //! [visualization]
    }
}

Rect disparidade::computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance){
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

