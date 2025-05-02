#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    cv::imwrite("test.png", image);
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
}