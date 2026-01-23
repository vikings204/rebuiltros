#include <cstdio>
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "cuAprilTags.h"
#include <chrono>

void loadImageToCuAprilTagsInput(const std::string& imagePath, cuAprilTagsImageInput_t& inputImage) {
  // Load image using OpenCV
  cv::Mat imgGs = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
  if (imgGs.empty()) {
    throw std::runtime_error("Failed to load image!");
  }

  cv::Mat img;
  cv::cvtColor(imgGs, img, cv::COLOR_GRAY2RGB);

  // Check if conversion worked
  if (img.type() != CV_8UC3) {
    throw std::runtime_error("Failed to convert to RGB!");
  }



  cudaMalloc(&inputImage.dev_ptr, img.rows * img.cols * sizeof(uchar3) * 8);
  cudaError_t memcpyErr = cudaMemcpy(inputImage.dev_ptr, img.data, img.rows * img.cols*sizeof(uchar3)*8, cudaMemcpyHostToDevice); // skeptical
  std::cout << "memcpy err: " << cudaGetErrorString(memcpyErr) << "\n";

  unsigned char* data = new unsigned char[img.rows * img.cols * sizeof(uchar3) * 8];
  cudaMemcpy(data, inputImage.dev_ptr, img.rows * img.cols * sizeof(uchar3) * 8, cudaMemcpyDeviceToHost);
  cv::Mat outimg(img.rows, img.cols, CV_8UC3, data);
  cv::imwrite("out.jpg", outimg);


  inputImage.width = img.cols;
  inputImage.height = img.rows;
  inputImage.pitch = img.cols*sizeof(uchar3);


}

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  printf("hello world apriltags package\n");

  cuAprilTagsImageInput_t inputImage;
  loadImageToCuAprilTagsInput("/home/team204/ros2_ws/image2.jpg", inputImage);


  cuAprilTagsCameraIntrinsics_t intrinsics{
    1991.333,
    1982.145,
    766.253,
    652.362,
  };
  cuAprilTagsHandle detector = nullptr;
  cudaStream_t stream = {};

  const int error = nvCreateAprilTagsDetector(&detector, 1600, 1304, 4, cuAprilTagsFamily::NVAT_TAG36H11, &intrinsics, 0.1651);
  std::cout << "create err code: " << error << "\n";
  auto streamErr = cudaStreamCreate(&stream);
  if (streamErr != 0) {
    std::cout << cudaGetErrorString(streamErr);
  }

  uint32_t num_detections;
  std::vector<cuAprilTagsID_t> tags(10);
  auto timeStart = std::chrono::high_resolution_clock::now();
  const int error2 = cuAprilTagsDetect(detector, &inputImage, tags.data(), &num_detections, 10, stream);
  auto timeEnd = std::chrono::high_resolution_clock::now();
  std::cout << "detect err code: " << error2 << "\n";

  std::chrono::duration<double> timeElapsed = timeEnd - timeStart;
  std::cout << "elapsed time for detection: " << timeElapsed.count() << " s\n";
  std::cout << "estimated fps: " << 1 / timeElapsed.count() << "\n";

  if (num_detections > 0) {
    for (auto t : tags) {
      if (t.id == 0) {
        continue;
      }
      std::cout << "id=" << t.id << " tx=" << t.translation[0] << " ty=" << t.translation[1] << " tz=" << t.translation[2] << "\n";
    }
  } else {
    std::cout << "no detections.\n";
  }

  cudaFree(inputImage.dev_ptr);
  cudaStreamDestroy(stream);
  cuAprilTagsDestroy(detector);

  return 0;
}
