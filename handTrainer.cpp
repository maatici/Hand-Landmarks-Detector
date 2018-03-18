#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    try
    {
        std::string hands_train_xml = "";
        std::string hands_test_xml = "";
        std::string model_name = "HandDetector.svm";

        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<rectangle> > hand_boxes_train, hand_boxes_test;

        load_image_dataset(images_train, hand_boxes_train, hands_train_xml);
        load_image_dataset(images_test, hand_boxes_test, hands_test_xml);

        upsample_image_dataset<pyramid_down<3> >(images_train, hand_boxes_train);
        upsample_image_dataset<pyramid_down<3> >(images_test,  hand_boxes_test);

        cout << "num training images: " << images_train.size() << endl;
        cout << "num testing images:  " << images_test.size() << endl;


        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
        image_scanner_type scanner;

        scanner.set_detection_window_size(80, 120);
        structural_object_detection_trainer<image_scanner_type> trainer(scanner);

        // Set this to the number of processing cores on your machine.
        trainer.set_num_threads(4);

        trainer.set_match_eps(0.4);

        trainer.set_c(11);

        trainer.be_verbose();

        trainer.set_epsilon(0.01);


        object_detector<image_scanner_type> detector = trainer.train(images_train, hand_boxes_train);

        cout << "training results: " << test_object_detection_function(detector, images_train, hand_boxes_train) << endl;

        cout << "testing results:  " << test_object_detection_function(detector, images_test, hand_boxes_test) << endl;


        serialize(model_name) << detector;
        cout << "Model is saved" << endl;

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

