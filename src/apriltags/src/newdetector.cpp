#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "frc971/orin/971apriltag.h"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/tag36h11.h"

int main(int argc, char ** argv)
{
    (void) argc;
    (void) argv;

    cv::VideoCapture cap{0, cv::CAP_V4L2};
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1600);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1304);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
    cap.set(cv::CAP_PROP_EXPOSURE, 75); // 0 to 100+
    cap.set(cv::CAP_PROP_GAIN, 0);
    cv::Mat frame;

    frc971::apriltag::CameraMatrix intrinsics{
        2030.2729812371426,
        748.1422029281408,
        2042.5411550248295,
        709.6692282439393,
    };
    frc971::apriltag::DistCoeffs distCoeffs{
        -0.286725755238781,
        4.185955041278353,
        0.016452211612830975,
        0.020408225747204543,
        -15.978664473424725,
    };

    constexpr float imageScalingFactor = 1.0;
    intrinsics.cx *= imageScalingFactor;
    intrinsics.cy *= imageScalingFactor;
    intrinsics.fx *= imageScalingFactor;
    intrinsics.fy *= imageScalingFactor;
    const int imageWidth = std::round(1600.0 * imageScalingFactor);
    const int imageHeight = std::round(1304.0 * imageScalingFactor);


    apriltag_detector_t *apriltagDetector = apriltag_detector_create();
    apriltag_detector_add_family_bits(apriltagDetector, tag36h11_create(), 1);
    apriltagDetector->nthreads = 6;
    apriltagDetector->wp = workerpool_create(apriltagDetector->nthreads);
    apriltagDetector->qtp.min_white_black_diff = 5;
    apriltagDetector->debug = false;

    frc971::apriltag::GpuDetector *gpuDetector = new frc971::apriltag::GpuDetector(imageWidth, imageHeight, apriltagDetector, intrinsics, distCoeffs);

    // cap >> frame;
    // cv::imwrite("stupid2.jpg", frame);
    // std::exit(0);

    // cv::Mat dummyImg = cv::imread("/home/team204/ros2_ws/dummy.jpg", cv::IMREAD_GRAYSCALE);
    // cv::cvtColor(dummyImg, frame, cv::COLOR_GRAY2RGB);

    while (true) {
        auto timeStart = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto timeCapture = std::chrono::high_resolution_clock::now();
        // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        //std::cout << "capt\n";
        if constexpr (imageScalingFactor < 1.0) {
            cv::resize(frame, frame, cv::Size(imageWidth, imageHeight), 0, 0, cv::INTER_AREA);
        }
        auto timeColorConvert = std::chrono::high_resolution_clock::now();

        gpuDetector->DetectGrayHost((unsigned char *)frame.ptr());
        const zarray_t *detections = gpuDetector->Detections();
        auto timeDetect = std::chrono::high_resolution_clock::now();

        if (zarray_size(detections) > 0) {
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);


                // const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>orientation(t.orientation);
                // const Eigen::Quaternion<float> q(orientation);
                // auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
                // double yaw = euler[1] * 180 / M_PI; // left-right
                // double pitch = euler[0] * 180 / M_PI; // up-down
                // double roll = euler[2] * 180 / M_PI; // turn
                //
                // // std::cout << "yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";
                // std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";
                std::cout << "id=" << det->id << " c(" << det->c[0] << "," << det->c[1] << ")\n";

                apriltag_detection_destroy(det);
            }
        }
        auto timeProcessing = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> timeToCapture = timeCapture - timeStart;
        std::chrono::duration<double> timeToColorConvert = timeColorConvert - timeCapture;
        std::chrono::duration<double> timeToDetect = timeDetect - timeColorConvert;
        std::chrono::duration<double> timeToProcess = timeProcessing - timeDetect;
        std::chrono::duration<double> timeTotal = timeProcessing - timeStart;
        std::cout << "capture: " << timeToCapture.count() << " s\n";
        std::cout << "colorconvert: " << timeToColorConvert.count() << " s\n";
        std::cout << "detect: " << timeToDetect.count() << " s\n";
        std::cout << "processing: " << timeToProcess.count() << " s\n";
        std::cout << "TOTAL: " << timeTotal.count() << " s\n";
        std::cout << "EST FPS: " << 1 / timeTotal.count() << "\n";
        std::cout << "num detections: " << zarray_size(detections) << "\n=====\n";
    }

    delete(gpuDetector);
    gpuDetector = nullptr;
    delete(apriltagDetector);
    apriltagDetector = nullptr;

    return 0;
}
