#include <dlib/image_processing.h>
#include <dlib/data_io.h>

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
    try
    {
        string trainxml = "";
        string testxml = "";
        string model = "Hand_9_Landmarks_Detector.dat";

        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > hands_train, hands_test;

        load_image_dataset(images_train, hands_train, trainxml);
        load_image_dataset(images_test, hands_test, testxml);

        shape_predictor_trainer trainer;

        trainer.set_oversampling_amount(20);
        trainer.set_nu(0.05);
        trainer.set_tree_depth(4);


        trainer.set_padding_mode(shape_predictor_trainer::padding_mode_t::bounding_box_relative);
        trainer.set_cascade_depth(15);
        trainer.set_num_threads(4);

        trainer.be_verbose();

        shape_predictor sp = trainer.train(images_train, hands_train);

        cout << "mean training error: "<< test_shape_predictor(sp, images_train, hands_train) << endl;

        cout << "mean testing error:  "<< test_shape_predictor(sp, images_test, hands_test) << endl;

        serialize(model) << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
