#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/svm_threaded.h>
#include "handCommonFunctions.hpp"

using namespace dlib;
using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
      if (argc != 2)
      {
           cout << "Give the path to the hand images directory as the argument to this program" << endl;
           return 0;
      }
      const string path = argv[1];
	

      string pathToLandmarkDetector = "models/Hand_9_Landmarks_Detector.dat";
      string pathToHandDetector = "models/HandDetector.svm";
      std::vector<string> files = std::vector<string>();

      getFilesInFolder(path,files);

      for (unsigned int i = 0;i < files.size();i++) {
          // Read Image
          cv::Mat im = cv::imread(files[i]);

          // Vector to store landmarks of all detected faces
          std::vector<full_object_detection> landmarksAll = detectLandmarks(pathToLandmarkDetector, pathToHandDetector, im);

          // Loop over all detected face rectangles
          for (int i = 0; i < landmarksAll.size(); i++)
          {
              // Draw landmarks on face
              drawLandmarks(im, landmarksAll[i]);
          }

          // Display image
          cv::imshow("Hand Landmarks", im);
          cv::waitKey(0);
      }

  return EXIT_SUCCESS;
}


