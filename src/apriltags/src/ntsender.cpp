#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"
#include <networktables/NetworkTableInstance.h>
#include <networktables/NetworkTable.h>
#include <networktables/DoubleTopic.h>
#include <networktables/IntegerTopic.h>
#include <eigen3/Eigen/Dense>
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
    auto nt_id_pub = table->GetIntegerTopic("id").Publish();
    auto nt_tx_pub = table->GetDoubleTopic("tx").Publish();
    auto nt_ty_pub = table->GetDoubleTopic("ty").Publish();
    auto nt_tz_pub = table->GetDoubleTopic("tz").Publish();
    auto nt_yaw_pub = table->GetDoubleTopic("yaw").Publish();
    auto nt_pitch_pub = table->GetDoubleTopic("pitch").Publish();
    auto nt_roll_pub = table->GetDoubleTopic("roll").Publish();
    nt_id_pub.SetDefault(0);
    nt_tx_pub.SetDefault(0);
    nt_ty_pub.SetDefault(0);
    nt_tz_pub.SetDefault(0);
    nt_yaw_pub.SetDefault(0);
    nt_pitch_pub.SetDefault(0);
    nt_roll_pub.SetDefault(0);

    cuAprilTagsCameraIntrinsics_t intrinsics{
        1991.333,
        1982.145,
        766.253,
        652.362,
    };
    cuAprilTagsHandle detector = nullptr;
    cudaStream_t stream = {};

    const int error = nvCreateAprilTagsDetector(&detector, 1600, 1304, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
    std::cout << "create error code: " << error << "\n";
    auto streamErr = cudaStreamCreate(&stream);
    if (streamErr != 0) {
        std::cout << cudaGetErrorString(streamErr);
    }

    uint32_t num_detections;
    std::vector<cuAprilTagsID_t> tags(22);

    const cudaError_t mallocErr = cudaMalloc(&imageInput.dev_ptr, 1304 * 1600 * sizeof(uchar3));
    std::cout << "malloc error: " << cudaGetErrorString(mallocErr) << "\n";

    // cap >> frame;
    // cv::imwrite("stupid2.jpg", frame);
    // std::exit(0);

    // cv::Mat dummyImg = cv::imread("/home/team204/ros2_ws/dummy.jpg", cv::IMREAD_GRAYSCALE);
    // cv::cvtColor(dummyImg, frame, cv::COLOR_GRAY2RGB);

    while (true) {
        auto timeStart = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto timeCapture = std::chrono::high_resolution_clock::now();
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        //std::cout << "capt\n";
        auto timeColorConvert = std::chrono::high_resolution_clock::now();

        const cudaError_t memcpyErr = cudaMemcpy(imageInput.dev_ptr, frame.data, 1304 * 1600 * sizeof(uchar3), cudaMemcpyHostToDevice);
        //std::cout << "memcpy error: " << cudaGetErrorString(memcpyErr) << "\n";
        imageInput.width = frame.cols;
        imageInput.height = frame.rows;
        imageInput.pitch = frame.cols*sizeof(uchar3);
        auto timeMemcpy = std::chrono::high_resolution_clock::now();

        const int error2 = cuAprilTagsDetect(detector, &imageInput, tags.data(), &num_detections, tags.capacity(), stream);
        //std::cout << "detect error code: " << error2 << "\n";
        auto timeDetect = std::chrono::high_resolution_clock::now();

        if (num_detections > 0) {
            for (auto t : tags) {
                if (t.id == 6 || t.id == 7 || t.id == 8 || t.id == 9 || t.id == 10 || t.id == 11 || t.id == 17 || t.id == 18 || t.id == 19 || t.id == 20 || t.id == 21 || t.id == 22) {
                    //std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " r0=" << t.orientation[0] << " r1=" << t.orientation[1] << " r2=" << t.orientation[2] << " r3=" << t.orientation[3] << " r4=" << t.orientation[4] << " r5=" << t.orientation[5] << " r6=" << t.orientation[6] << " r7=" << t.orientation[7] << " r8=" << t.orientation[8] << "\n";
                    const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>orientation(t.orientation);
                    const Eigen::Quaternion<float> q(orientation);
                    auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
                    double yaw = euler[1] * 180 / M_PI; // left-right
                    double pitch = euler[0] * 180 / M_PI; // up-down
                    double roll = euler[2] * 180 / M_PI; // turn

                    // std::cout << "yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";
                    //std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << "\n";

                    nt_id_pub.Set(t.id);
                    nt_tx_pub.Set(t.translation[0]);
                    nt_ty_pub.Set(t.translation[1]);
                    nt_tz_pub.Set(t.translation[2]);
                    nt_yaw_pub.Set(yaw);
                    nt_pitch_pub.Set(pitch);
                    nt_roll_pub.Set(roll);
                    break;
                }
            }
        } else {
            nt_id_pub.Set(0); // let java know that we lost the tag
            //nt_tx_pub.Set(0);
            //nt_ty_pub.Set(0);
            //nt_tz_pub.Set(0);
            //nt_yaw_pub.Set(0);
            //nt_pitch_pub.Set(0);
            //nt_roll_pub.Set(0);
            //std::cout << "no detections.\n";
        }
        auto timeProcessing = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> timeToCapture = timeCapture - timeStart;
        std::chrono::duration<double> timeToColorConvert = timeColorConvert - timeCapture;
        std::chrono::duration<double> timeToMemcpy = timeMemcpy - timeColorConvert;
        std::chrono::duration<double> timeToDetect = timeDetect - timeMemcpy;
        std::chrono::duration<double> timeToProcess = timeProcessing - timeDetect;
        std::chrono::duration<double> timeTotal = timeProcessing - timeStart;
        std::cout << "capture: " << timeToCapture.count() << " s\n";
        std::cout << "colorconvert: " << timeToColorConvert.count() << " s\n";
        std::cout << "memcpy: " << timeToMemcpy.count() << " s\n";
        std::cout << "detect: " << timeToDetect.count() << " s\n";
        std::cout << "processing: " << timeToProcess.count() << " s\n";
        std::cout << "TOTAL: " << timeTotal.count() << " s\n";
        std::cout << "EST FPS: " << 1 / timeTotal.count() << "\n";
        std::cout << "num detections: " << num_detections << "\n=====\n";
    }

  cudaFree(imageInput.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
