#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include "handCommonFunctions.hpp"

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    try
    {

        if (argc != 2)
        {
            cout << "Give the path to the hand images directory as the argument to this program" << endl;
            return 0;
        }
        string pathToDetector = "models/HandDetector.svm";
        string dir = argv[1];
        std::vector<string> files = std::vector<string>();

        getFilesInFolder(dir,files);

        for (unsigned int i = 0;i < files.size();i++) {

            // Read Image
            cv::Mat im = cv::imread(files[i]);

            // Detect hands boxes in the image
            std::vector<dlib::rectangle> handBoxes = detectHands(pathToDetector,im);
            cout << "Number of hands detected: " << handBoxes.size() << endl;

            for(int k=0; k< handBoxes.size();k++){
                // Display image
                drawBox(im,handBoxes[k]);
            }

            cv::imshow("Hand Box", im);
            cv::waitKey(0);

        }

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------


