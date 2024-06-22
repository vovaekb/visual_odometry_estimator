#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <Eigen/Core>

#include "OdometryEstimator.h"

// using fs = std::filesystem;
using namespace std::filesystem;
using namespace cv;

namespace cpp_practicing {
    using string_vector = std::vector<std::string>;
    using json = nlohmann::json;

    namespace {

        // float util_func1(const std::vector<float>& targets)
        // {
            
        // }

        float util_func1(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
            return 0.0;
        }

        Eigen::MatrixXf convertQuaternionToMatrix(OdometryEstimator::Rotation rotation)
        {
            Eigen::MatrixXf result;
            // ...
            return result;
        }
    }

    OdometryEstimator::OdometryEstimator(const std::string& frame_images_file_path, int fast_threshold) : 
            m_frames_files_path(frame_images_file_path), m_fast_threshold(fast_threshold) {

        m_frame_images.reserve(MAX_FRAMES_NUMBER);
    }

    void OdometryEstimator::run() {
        std::cout << "run visual odometry estimation" << std::endl;

        // Load folder with camera frames
        // Take a pair of sequential images (camera frames)
        // Process (undistort) them
        // Estimate features on first frame and track these to the second one
        // Use RANSAC to calcualate Essential matrix
        // Estimate pose (R, t) btw these frames
        // Apply scale information(optional)
        // Save pose to vector
        // Optional: build trajectory from list of poses
        
        loadFrameImages();

        m_last_camera_frame = m_frame_images[m_last_camera_frame_index];

        startOdometryEstimation();

    }

    void OdometryEstimator::loadImageMetadata() {
        std::cout << "load image metadata" << std::endl;
    }

    void OdometryEstimator::loadFrameImages() {
        std::cout << "load frame images" << std::endl;

        path dir_path = m_frames_files_path;

        std::vector<std::filesystem::path> frame_file_paths;
        std::copy(directory_iterator(dir_path), directory_iterator(), std::back_inserter(frame_file_paths));
        std::sort(frame_file_paths.begin(), frame_file_paths.end());

        for (auto&  file_path: frame_file_paths)
        {
            auto file_name = file_path.filename();
            // std::cout << "file " << file_path.filename() << ", " << file_path.extension() << std::endl;
            if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
                Mat image = imread(file_path, IMREAD_COLOR);
                cvtColor(image, image, COLOR_BGR2GRAY);
                m_frame_images.emplace_back(FrameSample {.file_name = file_name, .image_data = image});

                // auto image_size = image.size();
                // std::cout << "image_size: " << image_size.width << " x " << image_size.height << std::endl;

            }
        }

        // for (auto &&view : view_images)
        // {
        //     std::cout << "view image " << view.file_name << std::endl;
        // }
    }

    void OdometryEstimator::startOdometryEstimation() {
        // std::cout << "load frame images" << std::endl;

        while(m_last_camera_frame_index != m_frame_images.size() - 1)
        {
            if (m_last_camera_frame_index == 15) break;
            auto first_frame_idx = m_last_camera_frame_index;
            auto second_frame_idx = first_frame_idx + 1;
            // std::cout << "match pair with indices: " << first_frame_idx << ", " << second_frame_idx << std::endl;
            matchFramePair();
            m_last_camera_frame_index++;
        }
        
    }

    void OdometryEstimator::matchFramePair() {
        std::cout << "\nmatch two frame images" << std::endl;

        if (m_last_camera_frame_index == 0)
        {
            detectImageFeatures(m_last_camera_frame);
        }
        // std::cout << "keypoints in last frame (first frame): " << m_last_camera_frame.keypoints.size() << std::endl;

        auto new_frame = m_frame_images[m_last_camera_frame_index + 1];
        // detectImageFeatures(new_frame);
        // std::cout << "keypoints in new_frame (second frame): " << new_frame.keypoints.size() << std::endl;

        uchar_vector status;
        trackFeatures(new_frame, status);

        if (new_frame.keypoints.size() < m_minimum_track_features_number)
        {
            /* perform feature re-detection */
            detectImageFeatures(new_frame);
        }
        // Get transformation pose
        calculateTransformation(new_frame);
        m_last_camera_frame = new_frame;
        
    }

    void OdometryEstimator::detectImageFeatures(FrameSample& frame_image) {
        // std::cout << "detecting keypoints ...\n";
        keypoints_vector keypoints;
        FAST(frame_image.image_data,
            keypoints, 
            KeypointDetParameters::THRESHOLD, 
            KeypointDetParameters::NONMAX_SUPPRESSION
        );
        // std::cout << "keypoints number: " << keypoints.size() << std::endl;
        KeyPoint::convert(keypoints, frame_image.keypoints, std::vector<int>());

    }
    // void OdometryEstimator::estimateImageFeatures(FrameSample& frame_image) {}

    void OdometryEstimator::trackFeatures(FrameSample& frame_image, uchar_vector& status) {
        // std::cout << "track features" << std::endl;
        std::vector<float> err;
        cv::Size window_size = cv::Size(21, 21);
        auto term_criteria = cv::TermCriteria(
            cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
            TrackingParameters::TERMINATION_MAX_COUNT, 
            TrackingParameters::TERMINATION_EPSILON
        );
 
        cv::calcOpticalFlowPyrLK(
            m_last_camera_frame.image_data,
            frame_image.image_data,
            m_last_camera_frame.keypoints,
            frame_image.keypoints, status, err, window_size,
            TrackingParameters::MAX_PYRAMID_LEVEL, 
            term_criteria,
            TrackingParameters::MIN_EIGEN_THRESHOLD);

        // getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int index_correction = 0;
        for (size_t i = 0; i < status.size(); ++i)
        {
            auto pt = frame_image.keypoints.at(i - index_correction);
            if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
            {
                if ((pt.x < 0) || (pt.y < 0))
                {
                    status.at(i) = 0;
                }
                m_last_camera_frame.keypoints.erase(m_last_camera_frame.keypoints.begin() + i - index_correction);
                frame_image.keypoints.erase(frame_image.keypoints.begin() + i - index_correction);
                index_correction++;
                
            }
            
        }
        // std::cout << "keypoints in first frame (" << m_last_camera_frame.file_name << ") after filtering: " << m_last_camera_frame.keypoints.size() << std::endl;
        // std::cout << "keypoints in second frame (" << frame_image.file_name << ") after filtering: " << frame_image.keypoints.size() << "\n\n";
        
    }
        
    void OdometryEstimator::calculateTransformation(FrameSample& frame_image) {
        // Apply RANSAC to remove outliers
        std::cout << "Applying RANSAC to remove outliers ...\n";
        // recovering the pose and the essential matrix
        cv::Mat E, R, t, mask;
        E = cv::findEssentialMat(
            frame_image.keypoints,
            m_last_camera_frame.keypoints,
            calibration_params.focal,
            calibration_params.pp,
            cv::RANSAC,
            RansacParameters::PROBABILITY,
            RansacParameters::THRESHOLD,
            mask
        );
        cv::recoverPose(
            E, frame_image.keypoints,
            m_last_camera_frame.keypoints,
            R, t, 
            calibration_params.focal,
            calibration_params.pp,
            mask);
        for (size_t i = 0; i < R.rows; ++i)
        {
            for (size_t j = 0; j < R.cols; ++j)
            {
                std::cout << R.at<double>(i, j) << "\n";
            }
            std::cout << "\n";
            
        }
        
    }
    
    void OdometryEstimator::calculateNewPose() {

    }

    auto OdometryEstimator::getFrameSamples () const -> std::vector<FrameSample> {
        return m_frame_images;
    }

}