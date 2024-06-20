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
            m_frames_files_path(frame_images_file_path), {

        m_frame_images.reserve(MAX_FRAMES_NUMBER);
    }

    void OdometryEstimator::run() {
        std::cout << "run visual odometry estimation" << std::endl;

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

        uchar_vector status;
        trackFeatures(new_frame, status);

        if (new_frame.keypoints.size() < m_minimum_track_features_number)
        {
            /* perform feature re-detection */
            detectImageFeatures(new_frame);
        }
        m_last_camera_frame = new_frame;
        
    }

    void OdometryEstimator::detectImageFeatures(FrameSample& frame_image) {
        keypoints_vector keypoints;
        FAST(frame_image.image_data,
            keypoints, 
            KeypointDetParameters::THRESHOLD, 
            KeypointDetParameters::NONMAX_SUPPRESSION
        );
        KeyPoint::convert(keypoints, frame_image.keypoints, std::vector<int>());

    }

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
        
    }
        
    void OdometryEstimator::calculateTransformation(FrameSample& frame_image) {
        // Apply RANSAC to remove outliers
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
    
}