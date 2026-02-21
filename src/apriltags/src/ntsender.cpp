#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"
#include <frc/geometry/Quaternion.h>
#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Translation3d.h>
#include <frc/geometry/Transform3d.h>
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <networktables/DoubleTopic.h>
#include <frc/apriltag/AprilTagFieldLayout.h>
#include <units/length.h>
#include <Eigen/Dense> // do not put eigen3/Eigen/Dense, conflicts with wpi flavor of eigen lmao
#include <chrono>

int main(int argc, char ** argv)
{
    (void) argc;
    (void) argv;

    constexpr int imageWidth = 1600;
    constexpr int imageHeight = 1304;
    // constexpr int imageWidth = 1280;
    // constexpr int imageHeight = 720;
    // constexpr int imageWidth = 640;
    // constexpr int imageHeight = 480;

    cv::VideoCapture cap{0, cv::CAP_V4L2};
    cap.set(cv::CAP_PROP_FRAME_WIDTH, imageWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, imageHeight);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
    cap.set(cv::CAP_PROP_EXPOSURE, 75); // 0 to 100+
    cap.set(cv::CAP_PROP_GAIN, 0);
    cv::Mat frame;
    cuAprilTagsImageInput_t imageInput;

    auto inst = nt::NetworkTableInstance::GetDefault();
    auto table = inst.GetTable("datatable");
    inst.StartClient4("jetson");
    inst.SetServerTeam(204);
    inst.StartDSClient();
    auto nt_ts_pub = table->GetDoubleTopic("timestamp").Publish();
    auto nt_px_pub = table->GetDoubleTopic("px").Publish();
    auto nt_py_pub = table->GetDoubleTopic("py").Publish();
    auto nt_yaw_pub = table->GetDoubleTopic("yaw").Publish();
    auto nt_tags_pub = table->GetDoubleTopic("tags").Publish();
    auto nt_delay_pub = table->GetDoubleTopic("delay").Publish();
    nt_ts_pub.SetDefault(0);
    nt_px_pub.SetDefault(0);
    nt_py_pub.SetDefault(0);
    nt_yaw_pub.SetDefault(0);
    nt_tags_pub.SetDefault(0);
    nt_delay_pub.SetDefault(0);

    // camera matrix
    // [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
    // always calibrate at res
    cuAprilTagsCameraIntrinsics_t intrinsics{
        1997.8628505101703,
        1994.034269654484,
        712.7713941655713,
        593.5345604265101,
    };
    cuAprilTagsHandle detector = nullptr;
    cudaStream_t stream = {};

    const int createErr = nvCreateAprilTagsDetector(&detector, imageWidth, imageHeight, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
    if (createErr != 0) {
        std::cout << "create error code: " << cudaGetErrorString(cudaError_t(createErr)) << "\n";
    }
    auto streamErr = cudaStreamCreate(&stream);
    if (streamErr != 0) {
        std::cout << cudaGetErrorString(streamErr);
    }

    uint32_t num_detections;
    std::vector<cuAprilTagsID_t> tags(6);

    const cudaError_t mallocErr = cudaMalloc(&imageInput.dev_ptr, imageWidth * imageHeight * sizeof(uchar3));
    if (mallocErr != 0) {
        std::cout << "malloc error: " << cudaGetErrorString(mallocErr) << "\n";
    }

    auto layout = frc::AprilTagFieldLayout::LoadField(frc::AprilTagField::k2026RebuiltWelded);
    layout.SetOrigin(frc::AprilTagFieldLayout::OriginPosition::kBlueAllianceWallRightSide);

    // cap >> frame;
    // cv::imwrite("stupid2.jpg", frame);
    // std::exit(0);

    // cv::Mat dummyImg = cv::imread("/home/team204/ros2_ws/dummy.jpg", cv::IMREAD_GRAYSCALE);
    // cv::cvtColor(dummyImg, frame, cv::COLOR_GRAY2RGB);

    while (true) {
        // auto timeStart = std::chrono::high_resolution_clock::now();
        auto captureTime = std::chrono::high_resolution_clock::now();
        cap >> frame;
        // auto timeCapture = std::chrono::high_resolution_clock::now();
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // cv::flip(frame, frame, -1);
        //std::cout << "capt\n";
        // auto timeColorConvert = std::chrono::high_resolution_clock::now();

        const cudaError_t memcpyErr = cudaMemcpy(imageInput.dev_ptr, frame.data, imageWidth * imageHeight * sizeof(uchar3), cudaMemcpyHostToDevice);
        if (memcpyErr != 0) {
            std::cout << "memcpy error: " << cudaGetErrorString(memcpyErr) << "\n";
        }
        imageInput.width = frame.cols;
        imageInput.height = frame.rows;
        imageInput.pitch = frame.cols*sizeof(uchar3);
        // auto timeMemcpy = std::chrono::high_resolution_clock::now();

        const int detectErr = cuAprilTagsDetect(detector, &imageInput, tags.data(), &num_detections, tags.capacity(), stream);
        cuAprilTagsDetect(detector, &imageInput, tags.data(), &num_detections, tags.capacity(), stream);
        if (detectErr != 0) {
            std::cout << "detect error code: " << cudaGetErrorString(cudaError_t(detectErr)) << "\n";
        }
        // auto timeDetect = std::chrono::high_resolution_clock::now();

        if (num_detections > 0) {
            std::vector<frc::Pose3d> poses;
            std::vector<float> magnitudes;

            for (auto t : tags) {
                if (t.id == 0 || t.id > 32) {
                    // empty element or unknown tag
                    continue;
                }

                const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>orientation(t.orientation);
                const Eigen::Quaternion<float> q(orientation);
                auto euler = q.toRotationMatrix().canonicalEulerAngles(0, 1, 2);
                double yaw = euler[1] * 180 / M_PI; // left-right
                double pitch = euler[0] * 180 / M_PI; // up-down
                double roll = euler[2] * 180 / M_PI; // turn

                std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";

                auto rot = frc::Rotation3d(units::radian_t{euler[2]}, units::radian_t{euler[0]}, units::radian_t{-euler[1]});
                auto trl = frc::Translation3d(units::meter_t{-t.translation[2]}, units::meter_t{t.translation[0]}, units::meter_t{-t.translation[1]});
                poses.emplace_back(layout.GetTagPose(t.id).value().TransformBy(frc::Transform3d(trl, rot).Inverse()));
                magnitudes.push_back(sqrt(t.translation[0]*t.translation[0] + t.translation[1]*t.translation[1] + t.translation[2]*t.translation[2]));
            }

            if (poses.size() == 1) {
                // one tag, just send the pose
                auto p = poses[0].ToPose2d().RotateBy(frc::Rotation2d(units::radian_t{M_PI}));
                auto sendTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> secondsSinceCapture = sendTime - captureTime;

                std::cout << "tags=1 px=" << p.X().value() << " py=" << p.Y().value() << " yaw=" << p.Rotation().Degrees().value() << " time=" << secondsSinceCapture.count() << "s fps=" << 1.0/secondsSinceCapture.count() << "hz\n";
                nt_px_pub.Set(p.X().value());
                nt_py_pub.Set(p.Y().value());
                nt_yaw_pub.Set(p.Rotation().Degrees().value());
                nt_tags_pub.Set(1);
                nt_delay_pub.Set(secondsSinceCapture.count());
                nt_ts_pub.Set(sendTime.time_since_epoch().count());
            } else {
                // weighted average based on cam-to-tag distance
                double total_px = 0.0;
                double total_py = 0.0;
                double total_yaw = 0.0;
                double total_weight = 0.0;
                for (unsigned long i = 0; i < poses.size(); i++) {
                    auto p = poses[i].ToPose2d().RotateBy(frc::Rotation2d(units::radian_t{M_PI}));
                    double w = pow(M_E, -0.5 * magnitudes[i]);
                    total_px += p.X().value() * w;
                    total_py += p.Y().value() * w;
                    total_yaw += p.Rotation().Degrees().value() * w;
                    total_weight += w;
                }
                double px = total_px / total_weight;
                double py = total_py / total_weight;
                double yaw = total_yaw / total_weight;

                auto sendTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> secondsSinceCapture = sendTime - captureTime;
                std::cout << "tags=" << poses.size() << " px=" << px << " py=" << py << " yaw=" << yaw << " time=" << secondsSinceCapture.count() << "s fps=" << 1.0/secondsSinceCapture.count() << "hz\n";
                nt_px_pub.Set(px);
                nt_py_pub.Set(py);
                nt_yaw_pub.Set(yaw);
                nt_tags_pub.Set(poses.size());
                nt_delay_pub.Set(secondsSinceCapture.count());
                nt_ts_pub.Set(sendTime.time_since_epoch().count());
            }
        } else {
            nt_tags_pub.Set(0.0);
        }
        // auto timeProcessing = std::chrono::high_resolution_clock::now();

        // std::chrono::duration<double> timeToCapture = timeCapture - timeStart;
        // std::chrono::duration<double> timeToColorConvert = timeColorConvert - timeCapture;
        // std::chrono::duration<double> timeToMemcpy = timeMemcpy - timeColorConvert;
        // std::chrono::duration<double> timeToDetect = timeDetect - timeMemcpy;
        // std::chrono::duration<double> timeToProcess = timeProcessing - timeDetect;
        // std::chrono::duration<double> timeTotal = timeProcessing - timeStart;
        // std::cout << "capture: " << timeToCapture.count() << " s\n";
        // std::cout << "colorconvert: " << timeToColorConvert.count() << " s\n";
        // std::cout << "memcpy: " << timeToMemcpy.count() << " s\n";
        // std::cout << "detect: " << timeToDetect.count() << " s\n";
        // std::cout << "processing: " << timeToProcess.count() << " s\n";
        // std::cout << "TOTAL: " << timeTotal.count() << " s\n";
        // std::cout << "EST FPS: " << 1 / timeTotal.count() << "\n";
        // std::cout << "num detections: " << num_detections << "\n=====\n";
    }

  cudaFree(imageInput.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
