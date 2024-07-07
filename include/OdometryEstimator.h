/**
 * @file OdometryEstimator.h
 * @brief Class for estimating odometry of robot based on video stream from camera 
 */


// https://avisingh599.github.io/vision/monocular-vo/
// data: https://www.cvlibs.net/datasets/kitti/eval_odometry.php


#include <vector>
#include <memory>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <Eigen/Dense>

#include <nlohmann/json.hpp>


using namespace cv;
// using Mat = cv::Mat;

namespace cpp_practicing {
    
    const int MAX_FRAMES_NUMBER = 100;

    /**
     * @brief Pipeline for estimating odometry of robot based on video stream from camera
     */
    class OdometryEstimator
    {
    public:
        using string_vector = std::vector<std::string>;
        using view_matches_vector = std::vector<DMatch>;
        using uchar_vector = std::vector<uchar>;
        using point2f_vector = std::vector<cv::Point2f>;
        using keypoints_vector = std::vector<cv::KeyPoint>;

        /**
         * @brief Struct for representing Rotation quaternion
         */
        struct Rotation
        {
            float w;
            float x;
            float y;
            float z;
        };

        struct CalibrationData
        {
            float focal = 718.8560;
            cv::Point2d pp;

            CalibrationData() : pp (cv::Point2d(607.1928, 185.2157)) {}
        };

        struct TransformPose
        {
            using Ptr = std::shared_ptr<TransformPose>;
            // Rotation rotation;
            cv::Mat rotation;
            // std::vector<float> translation;
            cv::Mat translation;
        };

        struct ImageMetadata
        {
            CalibrationData calibration_data;
            TransformPose pose;
        };

        /**
         * @brief Struct for representing single image sample
         */
        struct FrameSample
        {
            std::string file_name;
            Mat image_data;
            point2f_vector keypoints;
            Mat descriptors;
            int inliers_number;

            bool operator==(const FrameSample& other) const {
                return file_name == other.file_name;
            }
        };

        /**
         * @brief Struct for Ransac parameters
         */
        struct RansacParameters {
            /**!< desirable level of confidence (probability) */
            static constexpr auto PROBABILITY = 0.999;
            /**!< Similarity threshold for filtering outliers */
            static constexpr auto THRESHOLD = 1.0;
        };

        /**
         * @brief Struct for FAST keypoint detector parameters
         */
        struct KeypointDetParameters {
            /**!< desirable level of confidence (probability) */
            static constexpr auto NONMAX_SUPPRESSION = 0.999;
            /**!< Similarity threshold for filtering outliers */
            static constexpr auto THRESHOLD = 20;
        };

        /**
         * @brief Struct for Tracking algorithm parameters
         */
        struct TrackingParameters {
            /**!< maximal pyramid level number */
            static constexpr auto MAX_PYRAMID_LEVEL = 3;
            /**!< minimum eigen value of a 2x2 normal matrix of optical flow */
            static constexpr auto MIN_EIGEN_THRESHOLD = 0.001;
            /**!< desired accuracy */
            static constexpr auto TERMINATION_EPSILON = 0.01;
            /**!< maximum number of iterations/elements */
            static constexpr auto TERMINATION_MAX_COUNT = 0.01;
        };
        
        OdometryEstimator(const std::string& frame_images_file_path, int fast_threshold);
        /**
         * @brief Start pipeline for estimating odometry  
         * */
        void run();

        auto getFrameSamples () const -> std::vector<FrameSample>;

        auto getLastCameraFrameIndex () const -> int;

        auto getLastCameraFrame () const -> FrameSample;
        
    private:
        /// last processed frame from camera
        FrameSample m_last_camera_frame;
        /// position of camera in the last frame
        TransformPose::Ptr m_last_camera_pose;
        /// images of frames from camera
        std::vector<FrameSample> m_frame_images;
        /// transformation poses of frames from camera
        std::vector<TransformPose::Ptr> frame_sequence_poses;
        /// feature detector
        // TransformPose result_pose;
        Eigen::MatrixXf result_pose_rotation;
        Eigen::Array33f camera_matrix;
        std::string m_query_metadata_file;
        /// file path to directory with camera frame images
        std::string m_frames_files_path;
        /// minimal required number of features for tracking
        int m_minimum_track_features_number = 1000;
        int min_hessian = 400;
        /// index of last processed frame from camera
        int m_last_camera_frame_index = 0;
        int m_fast_threshold = 20;
        bool m_fast_nonmax_suppression = true;
        CalibrationData calibration_params;
        RansacParameters ransac_params;
        
        void loadImageMetadata();
        void loadImagesPair();
        /**
         * @brief Load camera frame images from disk 
         * */
        void loadFrameImages();
        void startOdometryEstimation();
        /**
         * @brief Detect image features on the camera frame image 
         * */
        void detectImageFeatures(FrameSample& frame_image);
        // void estimateImageFeatures(FrameSample& frame_image);
        void matchFramePair();
        /**
         * @brief Track image features between neighboring camera frames using Lukas-Kanade algorithm 
         * */
        void trackFeatures(FrameSample& frame_image, uchar_vector& status);
        void calculateTransformation(FrameSample& frame_image);
        void calculateNewPose();
    };

} // namespace cpp_practicing