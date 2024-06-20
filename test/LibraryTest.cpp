#include <string>
#include <iostream>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

#include "OdometryEstimator.h"

using namespace std::filesystem;
using namespace cv;

namespace cpp_practicing {

    TEST(OdometryEstimatorTest, Initial) {
        auto fast_threshold = 5; // 10; // 20;
        std::string frame_images_path = "frames/image_0";
        OdometryEstimator odometry_estimator(frame_images_path, fast_threshold);
        odometry_estimator.run();
        ASSERT_TRUE(true);

        // path dir_path = result_path;

        // int files_number = 0;
        // for (const auto& file : directory_iterator(dir_path))
        // {
        //     ++files_number;
        // }
        // ASSERT_EQ(files_number, 3);
    }

    // TEST(OdometryEstimatorTest, Initial) {
    //     auto fast_threshold = 5; // 10; // 20;
    //     std::string frame_images_path = "frames/image_0";
    //     OdometryEstimator odometry_estimator(frame_images_path, fast_threshold);
    //     odometry_estimator.run();
    //     ASSERT_TRUE(true);
    // }
    
    TEST(OdometryEstimatorTest, LoadingFrameImages) {
        auto fast_threshold = 5;
        std::string frame_images_path = "frames/image_0";
        OdometryEstimator odometry_estimator(frame_images_path, fast_threshold);
        odometry_estimator.run();
        
        auto frame_images = odometry_estimator.getFrameSamples();
        EXPECT_TRUE(static_cast<int>(frame_images.size()) > 0);

        // path dir_path = result_path;

        // int files_number = 0;
        // for (const auto& file : directory_iterator(dir_path))
        // {
        //     ++files_number;
        // }
        // ASSERT_EQ(files_number, 3);
    }
}