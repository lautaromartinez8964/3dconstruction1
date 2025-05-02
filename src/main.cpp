#include <iostream>
#include <vector> // 包含一个标准库头文件，测试编译
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Hello from Single View Reconstruction C++ Project!" << std::endl;

    // 之后你的重建逻辑将从这里开始
    // 例如：加载数据、计算灭点、标定相机...

    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    cv::imwrite("test.png", image);
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    return 0;
    
}