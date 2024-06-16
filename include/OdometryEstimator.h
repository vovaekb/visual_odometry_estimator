/**
 * @file OdometryEstimator.h
 * @brief Class for estimating odometry of robot based on video stream from camera 
 */


#include <vector>
#include <memory>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <Eigen/Dense>


using namespace cv;

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

        struct TransformPose
        {
            using Ptr = std::shared_ptr<TransformPose>;
            Rotation rotation;
            std::vector<float> translation;
        };

        /**
         * @brief Struct for representing single image sample
         */
        struct FrameSample
        {
            std::string file_name;
            Mat image_data;
            std::vector<cv::Point2f> keypoints;
            Mat descriptors;
            int inliers_number;
        };
        
        OdometryEstimator(const std::string& frame_images_file_path, int fast_threshold);
        /**
         * @brief Start pipeline for estimating odometry  
         * */
        void run();
        
    private:
        /// last processed frame from camera
        FrameSample m_last_camera_frame;
        /// position of camera in the last frame
        TransformPose::Ptr m_last_camera_pose;
        /// images of frames from camera
        std::vector<FrameSample> m_frame_images;
        /// transformation poses of frames from camera
        std::vector<TransformPose::Ptr> frame_sequence_poses;
        // TransformPose result_pose;
        Eigen::MatrixXf result_pose_rotation;
        Eigen::Array33f camera_matrix;
        std::string m_query_metadata_file;
        /// file path to directory with camera frame images
        std::string m_frames_files_path;
        /// minimal required number of features for tracking
        int m_minimum_track_features_number = 1000;
        /// index of last processed frame from camera
        int m_last_camera_frame_index = 0;
        int m_fast_threshold = 20;
        bool m_fast_nonmax_suppression = true;
        
        void loadImagesPair();
        /**
         * @brief Load camera frame images from disk 
         * */
        void loadFrameImages();
        /**
         * @brief Start the pipeline for estimating odometry 
         * */
        void startOdometryEstimation();
        /**
         * @brief Detect image features on the camera frame image 
         * */
        void detectImageFeatures(FrameSample& frame_image);
        void matchFramePair();
        /**
         * @brief Track image features between neighboring camera frames using Lukas-Kanade algorithm 
         * */
        void trackFeatures(FrameSample& frame_image, std::vector<uchar>& status);
        void calculateTransformation();
        void calculateNewPose();
    };

} // namespace cpp_practicing