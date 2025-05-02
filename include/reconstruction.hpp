
#pragma once

#include<vector> //C++标准库：动态数组
#include<string> //C++标准库: 字符串
#include<map>    //C++标准库：映射(字典)
#include<utility> //C++ 标准库： 用于std:pair
#include<eigen3/Eigen/Dense> //Eigen核心功能（矩阵，向量）
#include<opencv4/opencv2/opencv.hpp> //OpenCv库（用于cv::Mat,cv::Vec3d等）
#include"json.hpp" //json库（用于JSON解析）

//使用Eigen命名空间
using namespace Eigen;
//使用nlohmann json命名空间
using json = nlohmann::json;

//定义结构体来存储从JSON加载的标注数据
struct LabelData {
    //存储平行线段：映射关系（标签名->一系列线段对）
    //每个点存储为齐次坐标(x,y,1)
    std::map<std::string,std::vector<std::pair<Vector3d,Vector3d>>> line2points;
    
    //图像中平面交线的公共点（后缀的h代表齐次 homogeneous)
    // 即存储三个主平面（通常是 XZ、YZ、XY 平面）的交点在图像上的齐次坐标 [x, y, 1]
    Vector3d intersection_point_h;

    //三个平面(通常对应XZ,YZ,XY的掩码图像) 掩码是二值图像，三张图，例如masks[0] 可能是 XZ 平面的掩码，其中值为 255 的像素表示属于 XZ 平面，值为 0 表示不属于。
    std::vector<cv::Mat> masks;

    //输入图像的尺寸(宽度x高度)
    cv::Size image_size;
    
    

};

//用于存储计算出的影消点和影消线信息
struct VanishingData{
    std::map<std::string, Vector3d> points; //将线段标签映射到对应的影消点(图像上的点)的齐次坐标[x,y,1]
    Matrix3d lines; //3x3矩阵，每一列对应一个平面的影消线
};

//用于存储最终的三维重建结果
struct ReconstructionResult {
    //存储重建出的所有三维点的坐标（点云）
    std::vector<Eigen::Vector3d> points_3d;

    //存储每个三维点对应的颜色 格式为OpenCV的BGR颜色
    std::vector<cv::Vec3b> colors;
};

// ---函数声明----

/* 1
功能：求解齐次线性方程组Ax=0 输入稀疏矩阵A（mxn，通常m≥n) 输出解向量x，即A的最小奇异值对应的右奇异向量
方法：对A进行SVD奇异值分解，V的最后一列即为x
*/
Eigen::VectorXd solveHomogeneousLinearSystem(const Eigen::MatrixXd& A);

/* 2
功能：加载并解析LabelMe格式的JSON文件，提取平行线、交点和平面掩码，作为后续重建的输入数据

*/
LabelData getLabel(const std::string& img_path,
                   const std::string& line_json_path,
                   const std::string& plane_json_path);

/*3
功能：根据标注的平行线计算影消点和影消线（地平线），为相机标定和平面估计提供基础
输入：包含平行线端点信息的LabelData结构体
输出：VanishingData结构体，包含各方向的影消点与三个主平面的影消线
方法：对同一标签的平行线，求其交点 影消线则通过影消点的叉积计算
*/

VanishingData calculateVanishingInfo(const LabelData& label_data);


/*4
功能：利用相互正交的影消点/线估计相机内参矩阵K，实现相机标定
输入：影消信息结构 假设正方形像素，零倾斜
输出：3x3相机内参矩阵K 
*/
Matrix3d calibrateCamera(const VanishingData& vanishing_data);

/*5
功能：根据相机内参K和图像上的二维齐次坐标点（影消点），计算其在相机坐标系下的三维射线方向 即将图像点转为三维射线 后续结合深度或平面交点可得具体三维坐标。
输入：相机内参矩阵K，图像上的齐次坐标点[x,y,1]^T
输出：归一化的三位方向向量d
方法：利用v =Kd 反得d = (K^-1)V/|（K^-1)/V|
*/
Vector3d calculate3DDirection(const Matrix3d& K, const Vector3d& p_2D_H);

/* 6
功能：计算场景中三个主平面的方程(法向量n和距离d),形式为nX+d = 0;
输入：相机内参矩阵K，影消信息，包含三个平面的交点
输出：4x3矩阵，每列为[nx,ny,nz,d]^T,对应一个平面 三个平面通常是XZ,YZ,XY平面
*/

Matrix<double, 4, 3> calculateScenePlanes(const Eigen::Matrix3d& K,
    const VanishingData& vanishing_data,
    const LabelData& label_data);

/*7
功能：对掩码区域内的像素点进行三维重建，生成点云
输入：内参矩阵K，场景平面方程矩阵(4x3)，输入图像(提取颜色),包含掩码与图像尺寸
*/
ReconstructionResult reconstruct3D(const Eigen::Matrix3d& K,
    const Eigen::Matrix<double, 4, 3>& Pi,
    const cv::Mat& img,
    const LabelData& label_data);

/*8
将重建的点云保存为 PLY 格式文件。
*/

bool createPlyOutput(const ReconstructionResult& result, const std::string& filename);



