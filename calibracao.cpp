#include "calibracao.h"

calibracao::calibracao(Mat& actualOne, Mat& actualTwo) : m_imageOne(actualOne), m_imageTwo(actualTwo) {
}

calibracao::~calibracao() {
    m_imageOne.release();
    m_imageTwo.release();
    img1.release();
    img2.release();
    gray1.release();
    gray2.release();
    m_imageOne.release();
    m_imageTwo.release();
}

void calibracao::iniciaCalibracaoCamera(){

    Size board_sz = Size(board_w, board_h);
    int board_n = board_w*board_h;
    
    for (int y = 0; y < board_h; y++) {
        for (int x = 0; x < board_w; x++) {
            obj.push_back(Point3f(y * squareSize, x * squareSize, 0));
        }
    }
        
    int success = 0;
    int k = 0;
    char key = 0;
    bool found1 = false;
    bool found2 = false;
    
    //Saves images for calibration
    while(success < numBoards){
        img1 = m_imageOne.clone();
        img2 = m_imageTwo.clone();
        
        cvtColor(img1, gray1, CV_BGR2GRAY);
        cvtColor(img2, gray2, CV_BGR2GRAY);
        
        found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        
        if(found1){
            cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray1, board_sz, corners1, found1);
        }
        
        if(found2){
            cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray2, board_sz, corners2, found2);
        }
        
        imshow("image1", gray1);
        imshow("image2", gray2);
        
        k = waitKey(10);
        key = (char) waitKey(10);
        
        if(found1 && found2){
            k = waitKey(0);
            key = (char) waitKey(0);
        }
        if(key == 27){
            break;
        }
        cout << "images amount: ";
        cout << success << endl;
        if(key == ' ' && found1 != 0 && found2 != 0){ // if(k == ' ' && found1 != 0 && found2 != 0)
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
            cout << "Stored corners " << endl;
            success++;
            
            if(success >= numBoards){
                break;
            }
        }
    }
    
    destroyAllWindows();
    
    //Calibrates and saves calibration data
    cout << "Starting Calibration..." << endl;
    
    CM1 = Mat(3, 3, CV_64FC1);
    CM2 = Mat(3, 3, CV_64FC1);
        
    stereoCalibrate(object_points, imagePoints1, imagePoints2, CM1, D1, CM2, 
            D2, img1.size(), R, T, E, F,
            CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST,
            cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
    
    FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
    fs1 << "CM1" << CM1;
    fs1 << "CM2" << CM2;
    fs1 << "D1" << D1;
    fs1 << "D2" << D2;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "E" << E;
    fs1 << "F" << F;
    
    cout << "Calibration completed" << endl;
    
    //Rectify and save the rectification data
    cout << "Starting Rectification..." << endl;
    
    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    fs1 << "R1" << R1;
    fs1 << "R2" << R2;
    fs1 << "P1" << P1;
    fs1 << "P2" << P2;
    fs1 << "Q" << Q;
    
    cout << Q << endl;
    
    fs1.release();
    cout << "Rectification completed" << endl;
    
    cout << "Applying Undistort..." << endl;
    
    Mat map1x, map1y, map2x, map2y;
    Mat imgU1, imgU2, imgRectify, img1Teste, img2Teste;
    
    initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);
    
    cout << "Undistort completed" << endl;
    
    while(1){
        img1Teste = m_imageOne.clone();
        img2Teste = m_imageTwo.clone(); 
        
        cvtColor(img1Teste, img1Teste, cv::COLOR_BGR2RGB);
        cvtColor(img2Teste, img2Teste, cv::COLOR_BGR2RGB);
       
        remap(img1Teste, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(img2Teste, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        
        //imshow("image1", img1Teste);
        //imshow("image2", img2Teste);
        
        //To display the rectified images
        imgRectify = Mat::zeros(imgU1.rows, imgU1.cols*2+10, imgU1.type());

        imgU1.copyTo(imgRectify(Range::all(), Range(0, imgU2.cols)));
        imgU2.copyTo(imgRectify(Range::all(), Range(imgU2.cols+10, imgU2.cols*2+10)));

        //If it is too large to fit on the screen, scale down by 2, it should fit.
        if(imgRectify.cols > 1920){
            resize(imgRectify, imgRectify, Size(imgRectify.cols/2, imgRectify.rows/2));
        }
        
        //To draw the lines in the rectified image
        for(int j = 0; j < imgRectify.rows; j += 16){
            Point p1 = Point(0,j);
            Point p2 = Point(imgRectify.cols*2,j);
            line(imgRectify, p1, p2, CV_RGB(255,0,0));
        }

        imshow("Rectified", imgRectify);
        
        k = waitKey(5);
        key = (char) waitKey(5);
        
        if(key==27){
            break;
        }
    }
}
