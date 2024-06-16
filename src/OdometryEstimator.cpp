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

using namespace std::filesystem;
using namespace cv;

namespace cpp_practicing {
    using string_vector = std::vector<std::string>;

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

    void OdometryEstimator::loadFrameImages() {
        path dir_path = m_frames_files_path;

        std::vector<std::filesystem::path> frame_file_paths;
        std::copy(directory_iterator(dir_path), directory_iterator(), std::back_inserter(frame_file_paths));
        std::sort(frame_file_paths.begin(), frame_file_paths.end());

        for (auto&  file_path: frame_file_paths)
        {
            auto file_name = file_path.filename();
            if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
                Mat image = imread(file_path, IMREAD_COLOR);
                m_frame_images.emplace_back(FrameSample {.file_name = file_name, .image_data = image});
            }
        }

    }

    void OdometryEstimator::startOdometryEstimation() {

        while(m_last_camera_frame_index != m_frame_images.size() - 1)
        {
            if (m_last_camera_frame_index == 15) break;
            auto first_frame_idx = m_last_camera_frame_index;
            auto second_frame_idx = first_frame_idx + 1;
            matchFramePair();
            m_last_camera_frame_index++;
        }
        
    }

    void OdometryEstimator::matchFramePair() {

        if (m_last_camera_frame_index == 0)
        {
            detectImageFeatures(m_last_camera_frame);
        }

        auto new_frame = m_frame_images[m_last_camera_frame_index + 1];

        std::vector<uchar> status;
        trackFeatures(new_frame, status);

        if (new_frame.keypoints.size() < m_minimum_track_features_number)
        {
            /* TODO: perform feature re-detection */
            detectImageFeatures(new_frame);
        }
        m_last_camera_frame = new_frame;
        
    }

    void OdometryEstimator::detectImageFeatures(FrameSample& frame_image) {
        std::vector<cv::KeyPoint> keypoints;
        FAST(frame_image.image_data, keypoints, m_fast_threshold, m_fast_nonmax_suppression);
        KeyPoint::convert(keypoints, frame_image.keypoints, std::vector<int>());

    }

    void OdometryEstimator::trackFeatures(FrameSample& frame_image, std::vector<uchar>& status) {
        std::vector<float> err;
        cv::Size window_size = cv::Size(21, 21);
        auto term_crit = cv::TermCriteria(
            cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 
            30, 0.01
        );

        auto min_eigen_threshold = 0.001;
        auto maximal_pyramid_level = 3; 
        cv::calcOpticalFlowPyrLK(
            m_last_camera_frame.image_data,
            frame_image.image_data,
            m_last_camera_frame.keypoints,
            frame_image.keypoints, status, err, window_size,
            maximal_pyramid_level, 
            term_crit, min_eigen_threshold);

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
        
    }
        
    void OdometryEstimator::calculateTransformation() {}
    
    void OdometryEstimator::calculateNewPose() {}

}