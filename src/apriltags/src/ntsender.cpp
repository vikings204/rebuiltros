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

    cv::VideoCapture cap{0, cv::CAP_V4L2};
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1600);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1304);
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
    auto nt_pz_pub = table->GetDoubleTopic("pz").Publish();
    auto nt_delay_pub = table->GetDoubleTopic("delay").Publish();
    nt_ts_pub.SetDefault(0);
    nt_px_pub.SetDefault(0);
    nt_py_pub.SetDefault(0);
    nt_pz_pub.SetDefault(0);
    nt_delay_pub.SetDefault(0);

    // camera matrix
    // [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
    cuAprilTagsCameraIntrinsics_t intrinsics{
        2050.45835471654,
        2050.2221293166149,
        744.36232048649231,
        624.70234622412681,
    };
    cuAprilTagsHandle detector = nullptr;
    cudaStream_t stream = {};

    constexpr float imageScalingFactor = 1.0;
    intrinsics.cx *= imageScalingFactor;
    intrinsics.cy *= imageScalingFactor;
    intrinsics.fx *= imageScalingFactor;
    intrinsics.fy *= imageScalingFactor;
    const int imageWidth = std::round(1600.0 * imageScalingFactor);
    const int imageHeight = std::round(1304.0 * imageScalingFactor);
    // intrinsics.cx = static_cast<float>(imageWidth) - intrinsics.cx;
    // intrinsics.cy = static_cast<float>(imageHeight) - intrinsics.cy;

    const int error = nvCreateAprilTagsDetector(&detector, imageWidth, imageHeight, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
    std::cout << "create error code: " << error << "\n";
    auto streamErr = cudaStreamCreate(&stream);
    if (streamErr != 0) {
        std::cout << cudaGetErrorString(streamErr);
    }

    uint32_t num_detections;
    std::vector<cuAprilTagsID_t> tags(6);

    const cudaError_t mallocErr = cudaMalloc(&imageInput.dev_ptr, imageWidth * imageHeight * sizeof(uchar3));
    std::cout << "malloc error: " << cudaGetErrorString(mallocErr) << "\n";

    Eigen::Matrix3d EDN_TO_NWU_matrix;
    EDN_TO_NWU_matrix << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    auto EDN_TO_NWU = frc::Rotation3d(EDN_TO_NWU_matrix);

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
        if constexpr (imageScalingFactor < 1.0) {
            cv::resize(frame, frame, cv::Size(imageWidth, imageHeight), 0, 0, cv::INTER_AREA);
        }
        // auto timeColorConvert = std::chrono::high_resolution_clock::now();

        const cudaError_t memcpyErr = cudaMemcpy(imageInput.dev_ptr, frame.data, 1304 * 1600 * sizeof(uchar3), cudaMemcpyHostToDevice);
        //std::cout << "memcpy error: " << cudaGetErrorString(memcpyErr) << "\n";
        imageInput.width = frame.cols;
        imageInput.height = frame.rows;
        imageInput.pitch = frame.cols*sizeof(uchar3);
        // auto timeMemcpy = std::chrono::high_resolution_clock::now();

        const int error2 = cuAprilTagsDetect(detector, &imageInput, tags.data(), &num_detections, tags.capacity(), stream);
        //std::cout << "detect error code: " << error2 << "\n";
        // auto timeDetect = std::chrono::high_resolution_clock::now();

        if (num_detections > 0) {
            std::vector<frc::Transform3d> transforms;
            std::vector<int> ids;

            for (auto t : tags) {
                if (t.id == 0) {
                    // empty element
                    continue;
                }

                //std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " r0=" << t.orientation[0] << " r1=" << t.orientation[1] << " r2=" << t.orientation[2] << " r3=" << t.orientation[3] << " r4=" << t.orientation[4] << " r5=" << t.orientation[5] << " r6=" << t.orientation[6] << " r7=" << t.orientation[7] << " r8=" << t.orientation[8] << "\n";
                const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>orientation(t.orientation);
                const Eigen::Quaternion<float> q(orientation);
                // auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
                auto euler = q.toRotationMatrix().canonicalEulerAngles(0, 1, 2);
                double yaw = euler[1] * 180 / M_PI; // left-right
                double pitch = euler[0] * 180 / M_PI; // up-down
                double roll = euler[2] * 180 / M_PI; // turn

                // std::cout << "yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";
                std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";

                // auto rot3dEDN = frc::Rotation3d(frc::Quaternion(q.w(), q.x(), q.y(), q.z()));
                auto rot3dEDN = frc::Rotation3d(units::radian_t{euler[2]}, units::radian_t{euler[0]}, units::radian_t{-euler[1]});
                // EDN_TO_NWU.unaryMinus().plus(rot.plus(EDN_TO_NWU));
                auto rot3dNWU = -EDN_TO_NWU + (rot3dEDN + EDN_TO_NWU);

                auto trl3dEDN = frc::Translation3d(units::meter_t{-t.translation[2]}, units::meter_t{t.translation[0]}, units::meter_t{-t.translation[1]});
                // auto trl3dEDN = frc::Translation3d(units::meter_t{t.translation[0]}, units::meter_t{t.translation[1]}, units::meter_t{t.translation[2]});
                auto trl3dNWU = trl3dEDN.RotateBy(EDN_TO_NWU);

                // auto best = frc::Transform3d(trl3dNWU, rot3dNWU);
                auto best = frc::Transform3d(trl3dEDN, rot3dEDN);
                transforms.emplace_back(best);
                ids.push_back(t.id);
            }

            if (transforms.size() == 1) {
                // one tag, just send the pose
                auto sendTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> secondsSinceCapture = sendTime - captureTime;
                auto t = layout.GetTagPose(ids[0]).value().TransformBy(transforms[0].Inverse());

                nt_px_pub.Set(t.X().value());
                nt_py_pub.Set(t.Y().value());
                nt_pz_pub.Set(t.Z().value());
                // also need t.rot.quat.wxyz
                std::cout << "px=" << t.X().value() << " py=" << t.Y().value() << " pz=" << t.Z().value() << "\n";
                nt_delay_pub.Set(secondsSinceCapture.count());
                nt_ts_pub.Set(sendTime.time_since_epoch().count());
            } else {
                // multitag, do sqpnp? or weighted avg based on distances
                std::cout << "multiple tags!! IDK WHAT TO DO\n";
            }

            break;
        } else {
            //nt_id_pub.Set(0); // let java know that we lost the tag
            //nt_tx_pub.Set(0);
            //nt_ty_pub.Set(0);
            //nt_tz_pub.Set(0);
            //nt_yaw_pub.Set(0);
            //nt_pitch_pub.Set(0);
            //nt_roll_pub.Set(0);
            //std::cout << "no detections.\n";
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
