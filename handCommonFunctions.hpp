#ifndef YAPAYZEKAOKULU_handCommon_HPP_
#define YAPAYZEKAOKULU_handCommon_HPP_

#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_loader/load_image.h>


using namespace dlib;
using namespace std;
using namespace cv;

int getFilesInFolder (string dir, std::vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error while opening " << dir << endl;
        return -1;
    }

    string extensions[] = {"jpg","jpeg","png","JPG","JPEG","PNG"};

    while ((dirp = readdir(dp)) != NULL) {
	 for(int k=0;k<6;k++)
    	 {
         	if(string(dirp->d_name).find(extensions[k]) != std::string::npos) {
            		files.push_back(dir + string(dirp->d_name));
         	}
    	 }     
    }
    closedir(dp);
    return 0;
}


void drawLandmarks(Mat &img, full_object_detection landmarks )
{
  for(int i = 0; i < landmarks.num_parts(); i++)
  {
    
    Point point = Point(landmarks.part(i).x(),landmarks.part(i).y());
    circle(img, point, 1, Scalar(0, 255, 0), 2, CV_AA);
    putText(img, std::to_string(i+1), point, FONT_HERSHEY_SIMPLEX, .5, Scalar(255, 191, 0), 1);
  }
}

void drawBox(Mat &img, dlib::rectangle box )
{

  Rect rect = Rect(box.left(),box.top(),box.right()-box.left(),box.bottom()-box.top());
  cv::rectangle(img,rect,Scalar(255,191,0),2,8,0);
}

Mat resizeImage(Mat img, int scaleFactor)
{
	Mat result;
	
	cv::resize(img,result,Size(), scaleFactor, scaleFactor, INTER_LINEAR );
	return result;

 
}

dlib::rectangle resizeBox(dlib::rectangle rect, int scaleFactor)
{
	dlib::rectangle result = dlib::rectangle(rect.left()/scaleFactor,rect.top()/scaleFactor,rect.right()/scaleFactor,rect.bottom()/scaleFactor);
	return result;
}


std::vector<dlib::rectangle> resizeBoxes(std::vector<dlib::rectangle> rects, int scaleFactor)
{
	
	for(int k=0; k< rects.size();k++)
	{
		rects[k] = resizeBox(rects[k],scaleFactor);
	}
	return rects;
}

std::vector<dlib::rectangle> detectHands(string pathToModel, Mat img)
{

    std::vector<dlib::rectangle> handBoxes;
    try
    {

        typedef scan_fhog_pyramid<pyramid_down<6>> image_scanner_type;
        object_detector<image_scanner_type> handDetector;
        deserialize(pathToModel) >> handDetector;

        // Convert OpenCV image format to Dlib's image format
        cv_image<bgr_pixel> dlibIm(img);

        handBoxes = handDetector(dlibIm);
        if(handBoxes.size()==0){
            cv::Mat imResize = resizeImage(img,3);
            cv_image<bgr_pixel> dlibImResize(imResize);
            handBoxes = handDetector(dlibImResize);
            handBoxes = resizeBoxes(handBoxes,3);
        }

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
    return handBoxes;
}

std::vector<full_object_detection> detectLandmarks(string pathToLandmarkDetector, string pathToHandDetector, Mat img)
{
	shape_predictor landmarkDetector;

    // Load the hand landmarks detector model
  	deserialize(pathToLandmarkDetector) >> landmarkDetector;
  	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

	// Load the Hand Detector model
	object_detector<image_scanner_type> handDetector;
  	deserialize(pathToHandDetector) >> handDetector;

	// Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(img);

	std::vector<dlib::rectangle> handBoxes = detectHands(pathToHandDetector,img);

	std::vector<full_object_detection> result;
	
	
	for (int i = 0; i < handBoxes.size(); i++)
    {
          // For every hand box, run landmarkDetector
          full_object_detection landmarks = landmarkDetector(dlibIm, handBoxes[i]);
          result.push_back(landmarks);
	}
	
	return result;
	
}
#endif
