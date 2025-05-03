#include <iostream>
#include <vector> // 包含一个标准库头文件，测试编译
#include <opencv4/opencv2/opencv.hpp>
#include "reconstruction.hpp"
#include <string>
#include <stdexcept> //异常处理
#include <filesystem> //C++17文件系统库

int main() {
     // --- 配置参数 ---

    // **关键点：相对路径的设置**
    // 当你使用标准的 CMake 工作流程时：
    // 1. 你在项目根目录 (包含 CMakeLists.txt, src, include, data, output 的那一层)
    // 2. 你创建一个 build 目录: `mkdir build && cd build`
    // 3. 你运行 cmake 配置: `cmake ..`
    // 4. 你运行 make 编译: `make`
    // 5. 可执行文件 `reconstructor` 会生成在 `build` 目录中。
    // 6. 当你在 `build` 目录中运行 `./reconstructor` 时，
    //    需要访问 `data` 和 `output` 目录，它们位于 `build` 目录的上一级。
    //    因此，我们使用相对路径 "../" 来表示上一级目录。

    const std::string base_folder = "../";  //相对于build目录的上一级
    const std::string data_folder = base_folder + "data/"; //指向../data/
    const std::string output_folder = base_folder + "output/"; //指向输出目录

    const std::string image_filename = data_folder + "images/chessboard.jpg";
    const std::string line_json_filename = data_folder + "labelme_data/chessboard_line.json";
    const std::string plane_json_filename = data_folder + "labelme_data/chessboard_plane.json";
    const std::string output_ply_filename = output_folder + "chessboard_3D_cpp.ply"; // 输出文件的完整路径
    
    //单视图重建流程
    try
    {
        std::cout<<"开始单视图重建..."<<std::endl;

       
           // 确保输出目录存在 (使用 C++17 文件系统库)
        if (!std::filesystem::exists(output_folder)) {
            std::cout << "输出目录不存在，正在创建: " << output_folder << std::endl;
            // create_directories 会创建所有必需的父目录
            if (!std::filesystem::create_directories(output_folder)) {
                 throw std::runtime_error("无法创建输出目录: " + output_folder);
            }
        } else {
            std::cout << "输出目录已存在: " << output_folder << std::endl;
        }

        //1.加载标注数据和图像信息
        std::cout << "步骤 1: 加载 JSON 标注和图像信息..." << std::endl;
        LabelData label_data = getLabel(image_filename, line_json_filename, plane_json_filename);
        std::cout << "  图像尺寸: " << label_data.image_size.width << " x " << label_data.image_size.height << std::endl;
        std::cout << "  已加载平行线段组数: " << label_data.line2points.size() << std::endl;
        std::cout << "  已加载交点。" << std::endl;
        std::cout << "  已生成平面掩码数量: " << label_data.masks.size() << std::endl;

         //2.计算影消点和影消线
        std::cout <<"步骤2：计算影消点与影消线"<<std::endl;
        VanishingData vanishing_data = calculateVanishingInfo(label_data);
        std::cout << "  已计算影消点点数量: " << vanishing_data.points.size() << std::endl;
        std::cout << "  已计算影消线数量: 3" << std::endl;

        //3.相机标定
              
        std::cout << "步骤 3: 相机标定..." << std::endl;
        Matrix3d K = calibrateCamera(vanishing_data);
        std::cout << "  计算得到的相机内参矩阵 K:\n" << K << std::endl;

        // 4. 计算场景平面方程
        std::cout << "步骤 4: 计算场景平面方程..." << std::endl;
        Matrix<double, 4, 3> Pi = calculateScenePlanes(K, vanishing_data, label_data);
        std::cout << "  计算得到的平面方程矩阵 Pi (4x3):\n" << Pi << std::endl;

        // 5. 重建三维点
        std::cout << "步骤 5: 重建三维点..." << std::endl;
        // 重新加载一次图像，专门用于提取颜色
        cv::Mat img_for_color = cv::imread(image_filename);
        if (img_for_color.empty()) {
             throw std::runtime_error("重新加载图像以提取颜色失败: " + image_filename);
        }
        ReconstructionResult recon_result = reconstruct3D(K, Pi, img_for_color, label_data);
        std::cout << "  已重建点数: " << recon_result.points_3d.size() << std::endl;

        // 6. 保存 PLY 输出文件
        std::cout << "步骤 6: 保存点云到 PLY 文件..." << std::endl;
        bool save_success = createPlyOutput(recon_result, output_ply_filename);

        if (save_success) {
            std::cout << "点云成功保存到: " << output_ply_filename << std::endl;
            std::cout << "重建完成！" << std::endl;
        } else {
            std::cerr << "保存点云失败。" << std::endl;
            return 1; // 返回错误码
        }


        
        
    }
    catch(const std::exception& e)
    {
        //捕获并打印流程中发生的任何标准异常
        std::cerr<<"\n---重建过程中发生错误---"<<std::endl;
        std::cerr<<"错误:"<<e.what()<<std::endl;
        return 1; //返回错误码
    }

    return 0; //程序成功结束 在c++中 返回零通常表示正常退出 返回非零值表示异常退出
    
}