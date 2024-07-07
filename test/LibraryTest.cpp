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

    }
    
    TEST(OdometryEstimatorTest, LoadingFrameImages) {
        auto fast_threshold = 5;
        std::string frame_images_path = "frames/image_0";
        OdometryEstimator odometry_estimator(frame_images_path, fast_threshold);
        odometry_estimator.run();
        
        auto frame_images = odometry_estimator.getFrameSamples();
        EXPECT_TRUE(static_cast<int>(frame_images.size()) > 0);
    }

    TEST(OdometryEstimatorTest, AllFrameImagesProcessed) {
        auto fast_threshold = 5;
        std::string frame_images_path = "frames/image_0";
        OdometryEstimator odometry_estimator(frame_images_path, fast_threshold);
        odometry_estimator.run();
        
        auto frame_images = odometry_estimator.getFrameSamples();
        auto frame_images_size = static_cast<int>(frame_images.size());
        auto last_camera_frame_index = odometry_estimator.getLastCameraFrameIndex();
        ASSERT_EQ(last_camera_frame_index, frame_images_size - 1);

        auto last_camera_frame = odometry_estimator.getLastCameraFrame();
        ASSERT_EQ(last_camera_frame, frame_images[frame_images_size - 1]);

    }
}