#include "reconstruction.hpp"
#include<iostream>
#include<stdexcept> //用于在出错时抛出异常
#include<vector>
#include<string>
#include<map>
#include<utility>
#include<iomanip> //用于PLY输出中的设置精度
#include<fstream>

//Eigen库功能包含
#include<Eigen/SVD>
#include<Eigen/Cholesky>

//OpenCV库特定功能的包含
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

using namespace Eigen;
using json = nlohmann::json;

/* 1
 * 功能：求解齐次线性方程组 Ax = 0。
 * 方法：使用 SVD 分解。返回与最小奇异值对应的右奇异向量。
 */
VectorXd solveHomogeneousLinearSystem(const Eigen::MatrixXd& A)
{
    if (A.rows() == 0 || A.cols() == 0)
    {
        throw std::runtime_error("输入给solveHomogeneousLinearSystem的矩阵A为空");
    }
    //计算SVD --我们需要右奇异向量V
    JacobiSVD<MatrixXd> svd(A,ComputeFullU|ComputeFullV);

    //检查SVD计算是否成功
    if (svd.info() != Success)
    {
        throw std::runtime_error("SVD计算失败");
    }

    //解是V的最后一列（对应最小的奇异值）
    return svd.matrixV().col(A.cols() - 1); //返回一个动态向量VectorXd
    
    
}

/* 2
 * 功能：加载 LabelMe JSON 数据, 提取平行线、交点和创建掩码。
 */
LabelData getLabel(const std::string& img_path, const std::string& line_json_path, const std::string& plane_json_path)
{
    LabelData result;

    // ---- 加载图像并获取尺寸 ----
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        throw std::runtime_error("加载图像失败: " + img_path);
    }
    result.image_size = img.size(); // 存储图像尺寸 (宽度 x 高度)

    // ---- 加载并解析标注线的 JSON 文件 ----
    std::ifstream line_file(line_json_path);
    if (!line_file.is_open()) {
        throw std::runtime_error("打开标注线 JSON 文件失败: " + line_json_path);
    }
    json data_line;
    try {
        line_file >> data_line; // 解析 JSON 文件内容
    } catch (const json::parse_error& e) {
        // 捕获并报告 JSON 解析错误
        throw std::runtime_error("解析标注线 JSON 失败: " + std::string(e.what()));
    }
    line_file.close(); // 关闭文件

    // ---- 处理线段标注 ----
    // 检查 JSON 文件是否包含 "shapes" 数组
    if (!data_line.contains("shapes") || !data_line["shapes"].is_array()) {
        throw std::runtime_error("无效的标注线 JSON 格式: 未找到 'shapes' 数组。");
    }

    // 定义预期的平行线标签
    std::vector<std::string> line_names = {"XZ_line", "YZ_line", "XY_line", "XZ_line'", "YZ_line'", "XY_line'"};

    // 遍历 "shapes" 数组中的每个标注形状
    for (const auto& shape : data_line["shapes"]) {
        // 检查形状是否包含有效的 "label" 和 "points"
        if (!shape.contains("label") || !shape["label"].is_string() ||
            !shape.contains("points") || !shape["points"].is_array()) {
            std::cerr << "警告: 跳过标注线 JSON 中的一个形状，因为它缺少或包含无效的 'label' 或 'points'。" << std::endl;
            continue; // 跳过这个形状
        }
        std::string label = shape["label"];          // 获取标签
        const auto& points_json = shape["points"]; // 获取点集

        // 检查是否是已知的平行线标签
        bool is_parallel_line = false;
        for (const auto& name : line_names) {
            if (label == name) {
                is_parallel_line = true;
                break;
            }
        }

        if (is_parallel_line) {
            // 检查平行线的点格式是否正确 (应该是一个包含两个点的数组)
            if (points_json.size() != 2 || !points_json[0].is_array() || points_json[0].size() != 2 ||
                !points_json[1].is_array() || points_json[1].size() != 2 ||
                !points_json[0][0].is_number() || !points_json[0][1].is_number() ||
                !points_json[1][0].is_number() || !points_json[1][1].is_number()) {
                std::cerr << "警告: 跳过平行线 '" << label << "'，因为其 'points' 格式无效 (期望 [[x1, y1], [x2, y2]])。" << std::endl;
                continue;
            }
            // 提取点坐标 (x, y)，并转换为齐次坐标 (y, x, 1) 以匹配 Python 中 reversed() 的逻辑
            double x1 = points_json[0][0]; double y1 = points_json[0][1];
            double x2 = points_json[1][0]; double y2 = points_json[1][1];
            Vector3d p1_h(y1, x1, 1.0); // 齐次坐标 (y, x, 1)
            Vector3d p2_h(y2, x2, 1.0);
            // 将这对点添加到对应标签的列表中
            result.line2points[label].push_back({p1_h, p2_h}); // 使用 C++17 的结构化绑定 {p1_h, p2_h}

        } else if (label == "intersection") {
             // 检查交点的点格式是否正确 (应该是一个只包含一个点的数组)
            if (points_json.size() != 1 || !points_json[0].is_array() || points_json[0].size() != 2 ||
                !points_json[0][0].is_number() || !points_json[0][1].is_number()) {
                 std::cerr << "警告: 跳过交点，因为其 'points' 格式无效 (期望 [[x, y]])。" << std::endl;
                 continue;
            }
             // 提取点坐标 (x, y)，并转换为齐次坐标 (y, x, 1)
            double x = points_json[0][0]; double y = points_json[0][1];
            result.intersection_point_h << y, x, 1.0; // 使用 Eigen 的逗号初始化器赋值
        }
        // 可以添加对其他标签类型的处理（如果需要）
    }
    // 检查是否成功找到了交点
     if (result.intersection_point_h.isZero(1e-9)) { // 使用 isZero 检查向量是否接近零向量
        throw std::runtime_error("未在标注线 JSON 中找到或解析有效的交点 'intersection'。");
    }


    // ---- 加载并解析标注平面的 JSON 文件 ----
    std::ifstream plane_file(plane_json_path);
     if (!plane_file.is_open()) {
        throw std::runtime_error("打开标注平面 JSON 文件失败: " + plane_json_path);
    }
    json data_plane;
    try {
        plane_file >> data_plane;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("解析标注平面 JSON 失败: " + std::string(e.what()));
    }
    plane_file.close();

    // ---- 根据平面标注创建掩码 ----
    if (!data_plane.contains("shapes") || !data_plane["shapes"].is_array()) {
        throw std::runtime_error("无效的标注平面 JSON 格式: 未找到 'shapes' 数组。");
    }

    // 定义我们期望的平面标签，并将它们映射到掩码的索引
    std::map<std::string, int> plane_label_to_index = {
        {"XZ_plane", 0}, // 假设 XZ 平面对应第一个掩码 (索引 0)
        {"YZ_plane", 1}, // 假设 YZ 平面对应第二个掩码 (索引 1)
        {"XY_plane", 2}  // 假设 XY 平面对应第三个掩码 (索引 2)
    };
    result.masks.resize(3); // 为 3 个掩码分配空间
    for (int i = 0; i < 3; ++i) {
        // 初始化掩码为全黑图像 (所有像素值为 0)
        // 尺寸使用图像的高和宽，类型为 CV_8UC1 (8位无符号单通道灰度图)
        result.masks[i] = cv::Mat::zeros(result.image_size.height, result.image_size.width, CV_8UC1);
    }

    // 遍历平面标注 JSON 中的 "shapes"
    for (const auto& shape : data_plane["shapes"]) {
         // 检查形状的有效性
         if (!shape.contains("label") || !shape["label"].is_string() ||
            !shape.contains("points") || !shape["points"].is_array() || shape["points"].empty()) {
            std::cerr << "警告: 跳过标注平面 JSON 中的一个形状，因为它缺少或包含无效的 'label' 或 'points'。" << std::endl;
            continue;
        }
        std::string label = shape["label"]; // 获取标签

        // 检查标签是否是我们关注的平面标签之一
        if (plane_label_to_index.count(label)) { // count() 比 find() 更简洁地检查 key 是否存在
            int mask_index = plane_label_to_index[label]; // 获取对应的掩码索引
            std::vector<cv::Point> polygon_points; // OpenCV 使用 Point(x, y) 表示点

            // 遍历形状的点集，创建多边形顶点列表
            for (const auto& point_json : shape["points"]) {
                // 检查点格式是否为 [number, number]
                if (!point_json.is_array() || point_json.size() != 2 || !point_json[0].is_number() || !point_json[1].is_number()) {
                     std::cerr << "警告: 跳过平面 '" << label << "' 多边形中的无效点。" << std::endl;
                     polygon_points.clear(); // 如果有无效点，则使该多边形无效
                     break; // 不再处理这个形状的其他点
                }
                 // 提取 (x, y) 坐标，并添加到多边形点列表
                 // 使用 std::round 将 double 转换为 int 作为像素坐标
                polygon_points.emplace_back(cv::Point(std::round(point_json[0].get<double>()), std::round(point_json[1].get<double>())));
            }

            // 如果成功提取了多边形的顶点
            if (!polygon_points.empty()) {
                 // cv::fillPoly 需要一个 C 风格的点数组或 vector<vector<Point>>
                std::vector<std::vector<cv::Point>> polygons = {polygon_points};
                // 在对应的掩码图像上填充该多边形区域，颜色为白色 (255)
                cv::fillPoly(result.masks[mask_index], polygons, cv::Scalar(255));
            }
        }
    }

    return result; // 返回包含所有解析数据的 LabelData 结构
   
}

/*3
*功能：根据标注的平行线计算影消点和影消线
*/

VanishingData calculateVanishingInfo(const LabelData& label_data)
{
    //创建VanishingData结构体vd，包含points影消点和lines影消线
    VanishingData vd; 
    
    //---计算影消点---
    //遍历label_data中存储的每一组平行线
    for (const auto& pair:label_data.line2points)
    {
        const std::string& label = pair.first; //当前平行线组的标签
        const auto& segments = pair.second; //该组包含的所有线段

        //至少需要两条平行线才能计算影消点
        if (segments.size() < 2)
        {
            std::cerr << "警告: 计算灭点需要至少两条平行线，标签 '" << label << "' 只有 " << segments.size() << " 条。跳过。" << std::endl;
            continue; //继续遍历下一条线
        }

        //构建矩阵A 用于求解Ax = 0 其中x是影消点
        //A的每一行是对应线段的直线方程系数 l = [a,b,c]^T; 直线方程由线段的起点p1和终点p2的叉积确定
        // li = p1xp2 p1=[x1,y1,1]^T p2 = [x2,y2,1]^T
        MatrixXd A(segments.size(),3);
        for (size_t i = 0; i < segments.size(); ++i)
        {
            //直线方程 l = p1xp2
            Vector3d line_eq = segments[i].first.cross(segments[i].second);
            //检查线段是否退化(两点重合)
            if (line_eq.norm() < 1e-9) //线段的模是否小于一个很小的数 如果是代表重合了
            {
                throw std::runtime_error("发现退化线段，标签"+label+",请检查标注");
            }
            A.row(i) = line_eq.transpose(); //将直线方程系数作为行向量存入A
            
        }

        //影消点是齐次线性方程Ax = 0的解 使用解齐次方程函数
        Vector3d vp_h = solveHomogeneousLinearSystem(A);

        //归一化齐次坐标(使最后一个分量为1)
        if(std::abs(vp_h(2) < 1e-9))
        {
            // 如果最后一个分量接近于 0，表示灭点在无穷远处（即图像中的平行线）
             // 这在透视投影中通常不应该发生，除非标注的线在图像上就是平行的
             std::cerr << "警告: 标签 '" << label << "' 的灭点似乎在无穷远处 (z≈0)。请检查标注。" << std::endl;
             // 可以选择报错，或者存储未归一化的点，或者尝试加一个微小量避免除零
             vp_h /= (vp_h(2) + 1e-12); //加微小量避免除零，但可能导致数值很大
        }
        else
        {
            vp_h /= vp_h(2); //正常归一化，使齐次坐标的第三个值为1
        }

        //将计算出的影消点存入结果的影消信息结构体
        vd.points[label] = vp_h;
    }

        //检查是否所有必需的影消点都已计算出来(XZ,YZ,XY) 即最终只有六个影消点（一个平面两组平行线段，两个点)
    std::vector<std::string> required_vp_labels = {"XZ_line", "YZ_line", "XY_line", "XZ_line'", "YZ_line'", "XY_line'"};
    for(const auto& req_label : required_vp_labels) {
        if (vd.points.find(req_label) == vd.points.end()) {
            // find() 返回 end() 表示未找到
            throw std::runtime_error("未能计算出必需的灭点: " + req_label);
        }
        
        
    }
    
    

    //----计算影消线----
    //影消线是对应平面上两个影消点的连线(通过叉积计算)
    try
    {
        //使用at()访问map元素，如果key不存在会抛出out_of_range异常
        vd.lines.col(0) = vd.points.at("XZ_line").cross(vd.points.at("XZ_line'"));
        vd.lines.col(1) = vd.points.at("YZ_line").cross(vd.points.at("YZ_line'"));
        vd.lines.col(2) = vd.points.at("XY_line").cross(vd.points.at("XY_line'"));
    }
    catch(const std::out_of_range& e)
    {
        throw std::runtime_error("缺少计算灭线所需的灭点。");
    }

    //归一化影消线方程(可选，但有助于数值稳定性)
    for (int i = 0; i < 3; ++i)
    {
        double norm = vd.lines.col(i).norm(); //计算向量的模
        if (norm > 1e-9)
        {
            vd.lines.col(i) /= norm; //向量除以其模进行归一化
        }
        else  
        {
             // 如果模为 0，说明影消线计算有问题（可能是影消点重合或共线）
             throw std::runtime_error("计算出的灭线 " + std::to_string(i) + " 范数接近零。灭点可能重合或共线。");
        }
        
    }

    return vd;
    
    
}

/*4
* 功能：利用相互正交的影消点估计相机内参矩阵K
* 假设：零倾斜、方形像素·
*/
Matrix3d calibrateCamera(const VanishingData& vanishing_data)
{
    //获取三个相互正交方向的影消点
    // 假设使用不带 ' 的标签 ("XZ_line", "YZ_line", "XY_line") 代表主轴方向
    // 注意：哪个标签对应哪个轴 (X, Y, Z) 取决于标注时的约定
    Vector3d v1,v2,v3;
    try {
        v1 = vanishing_data.points.at("XZ_line"); // 可能对应 X 或 Z 方向灭点
        v2 = vanishing_data.points.at("YZ_line"); // 可能对应 Y 方向灭点
        v3 = vanishing_data.points.at("XY_line"); // 可能对应 Z 或 X 方向灭点
    } catch (const std::out_of_range& oor) {
        throw std::runtime_error("缺少用于标定的主灭点 (XZ_line, YZ_line, XY_line)。");
    }

    //构建约束矩阵A
    //w(omega) = (KK^T)-1 两组平行线正交时，对应的影消点v1^Twv2 = 0
    //加上零倾斜和方形像素两个条件后，w = [w1,0,w4|0,w1,w5|w4,w5,w6]
    //三组平行线两两正交，v1^T wv 2 = 0 | v1^T w v3 =0 | v2^T w v3 = 0;
    //可以构建Aw = 0（具体见grok)
    //三行分别是vp1和vp2，vp1和vp3，vp2和vp3正交 每行的系数对应[w1,w4,w5,w6]
    MatrixXd A(3,4);
    A.row(0) << v1(0)*v2(0) + v1(1)*v2(1),   v1(0)*v2(2) + v1(2)*v2(0),   v1(1)*v2(2) + v1(2)*v2(1),   v1(2)*v2(2);
    A.row(1) << v1(0)*v3(0) + v1(1)*v3(1),   v1(0)*v3(2) + v1(2)*v3(0),   v1(1)*v3(2) + v1(2)*v3(1),   v1(2)*v3(2);
    A.row(2) << v2(0)*v3(0) + v2(1)*v3(1),   v2(0)*v3(2) + v2(2)*v3(0),   v2(1)*v3(2) + v2(2)*v3(1),   v2(2)*v3(2);
    
    //调用SVD函数 求解Aw = 0 返回的w = [w1,w4,w5,w6]^T 是齐次解（相差一个尺度因子)
    VectorXd w = solveHomogeneousLinearSystem(A);

    //构建w（Ω 矩阵） w = (KK^T)^-1
    Matrix3d Omega;
    Omega<<w(0),0.0,w(1),  //w1 0 w4
           0.0,w(0),w(2),  //w1,w5
           w(1),w(2),w(3); //w4 w5 w6

    //根据w6归一化（因为摄像机内参矩阵K 右下角元素为1）
    if (std::abs(Omega(2,2)) < 1e-9) {
        throw std::runtime_error("无法归一化 Omega：右下角元素 w6 接近零。");
    }
    Omega /= Omega(2, 2); // 除以 w6

    //计算KK^T KK^T = w^-1 并归一化右下角元素为1
    Matrix3d Omega_inv;
    bool invertible; //Eigen自动检查矩阵是否可逆
    double det; //行列式值
    Omega.computeInverseAndDetWithCheck(Omega_inv,det,invertible);
    if (!invertible) {
        throw std::runtime_error("Omega 矩阵奇异，无法计算逆矩阵用于 K K^T。");
    }
    
    if (std::abs(Omega_inv(2,2)) < 1e-9) {
        throw std::runtime_error("无法归一化 K K^T = Omega_inv：右下角元素接近零。");
    }
    Matrix3d KKt = Omega_inv / Omega_inv(2, 2);

    //从KK^T提取内参：fx，fy，cx，cy 方形像素，fx = fy
    double cx = KKt(0, 2);
    double cy = KKt(1, 2);

    double f_squared_plus_cy_squared = KKt(1, 1);
    double f_squared = f_squared_plus_cy_squared - cy * cy;
    if (f_squared < 0) {
    double f_squared_plus_cx_squared = KKt(0, 0);
    f_squared = f_squared_plus_cx_squared - cx * cx;
    if (f_squared < 0) {
        throw std::runtime_error("无法计算焦距：f^2 为负数。检查 Omega 或灭点计算。");
    }
    }
    if (f_squared < 1e-9) {
    throw std::runtime_error("计算出的焦距平方 f^2 接近零。");
    }
    double f = std::sqrt(f_squared);

    //构建K
    Matrix3d K;
    K <<f,0.0,cx,
        0.0,f,cy,
        0.0,0.0,1.0;
    K /= K(2,2);
    return K;
}

/* 5
*功能：根据相机内参K和图像上的2D齐次坐标点p_2D_H (y,x,1),计算其在相机坐标系下的3D方向向量
       即输出从相机坐标系原点到该点的射线方向
*/
Vector3d calculate3DDirection(const Matrix3d& K, const Vector3d& p_2D_H)
{
    //反投影，从图像2d点计算方向向量
    Vector3d direction = K.inverse() * p_2D_H; //算出来的3D点也就是方向向量

    //归一化方向向量
    direction.normalize();
    return direction;
}


/* 6
 * 功能：利用影消线lh和摄像机内参矩阵K计算场景中三个主平面 (XZ, YZ, XY) 的方程 Pi = [n; d]。
 * 平面方程形式为 n^T * X + d = 0，其中X=[X,Y,Z]^T是3D点 或齐次形式 Π^T * X_H = 0。 X_H = [X,Y,Z,1]^T是齐次坐标
 * 输出的4x3矩阵，每列是一个平面的参数[nx,ny,nz,d]^T 即ax+by+cz+d=0中的[a,b,c,d]^T
 */
Matrix<double, 4, 3> calculateScenePlanes(const Eigen::Matrix3d& K, const VanishingData& vanishing_data, const LabelData& label_data)
{
    //---计算平面法向量---
    Matrix3d N = K.transpose() * vanishing_data.lines; //n = K^T*lh 平面法向量 = K的转置 × 影消线
    Matrix3d Unit_N;

    //平面法向量归一化
    for (int i = 0; i<3; ++i)
    {
        double norm = N.col(i).norm(); //计算法向量矩阵每一列（每个平面的法向量）的模长
        if (norm < 1e-9)
        {
            throw std::runtime_error("计算出的平面法向量" + std::to_string(i) +"范数接近0");
        }
        Unit_N.col(i) = N.col(i)/norm; //Unit_n是3x3矩阵，每列是一个平面的单位法向量
    }

    //---计算平面到相机中心的距离d---
    //1.获取图像中三个平面交点的齐次坐标(y,x,1)
    Vector3d intersect_2d_h = label_data.intersection_point_h;
    //2.计算该焦点对应的相机坐标系下的3D方向向量
    Vector3d intersect_3d_dir = calculate3DDirection(K, intersect_2d_h);

    //假设交点的深度为1（即相机坐标系的原点在方向向量上的深度为1） 三个平面交点（相机坐标系原点）为X_intersect
    Vector3d X_intersect = 1.0*intersect_3d_dir;

    //计算平面到原点的距离d 平面:ni^T X_intersect + di = 0; di = -ni^T X_intersect
    Vector3d D = -Unit_N.transpose() * X_intersect; //D=[d0,d1,d2]^T,每个元素是对应平面的距离

    //---构建平面参数矩阵Pi（4x3）---
    //每列代表一个平面的参数[nx,ny,nz,d]^T
    Matrix<double,4,3>Pi;
    Pi.block<3, 3>(0, 0) = Unit_N;    // Pi 的前 3 行是单位法向量 Unit_N(左上角3x3的子矩阵)
    Pi.row(3) = D.transpose(); // Pi 的第 4 行是距离 d (需要转置 D)



    
}

/*7
功能：对掩码区域内的像素点进行三维重建，生成点云
输入：内参矩阵K，场景平面方程矩阵(4x3)，输入图像(提取颜色),包含掩码与图像尺寸
*/
ReconstructionResult reconstruct3D(const Eigen::Matrix3d& K, 
    const Eigen::Matrix<double, 4, 3>& Pi, 
    const cv::Mat& img, 
    const LabelData& label_data)
{
    ReconstructionResult result; //存储重建结果

    //检查输入图像和掩码的有效性
    if (img.empty() || img.rows != label_data.image_size.height || img.cols != label_data.image_size.width)
    {
        throw std::runtime_error("用于重建的输入图像为空或尺寸与标注数据不匹配。");
    }
    if(label_data.masks.size() != 3)
    {
        throw std::runtime_error("LabelData 中应包含 3 个掩码用于重建。");
    }

    Matrix3d K_inv = K.inverse();//预计算K的逆矩阵

    //遍历三个平面
    for (int axis = 0; axis<3; ++axis)
    {
        const cv::Mat& mask = label_data.masks[axis]; //获取当前平面的掩码 类型为CV_8UC1 值为0或255
        if (mask.empty() || mask.rows != img.rows || mask.cols != img.cols || mask.type() != CV_8UC1) {
            std::cerr << "警告: 轴 " << axis << " 的掩码无效或格式错误 (期望 CV_8UC1)。跳过此平面。" << std::endl;
            continue;
        }

        //获取当前平面的参数
        Vector4d plane_params = Pi.col(axis); 
        Vector3d n = plane_params.head<3>(); //法向量 n = [nx,ny,nz]^T
        double d = plane_params(3);

        //遍历图像的每个像素(r = row, c = column)
        for (int r = 0; r < img.rows; ++r) //r对应y坐标
        {
            for (int c = 0; c < img.cols; ++c) //c对应x坐标
            {
                if (mask.at<uchar>(r,c) > 0) //检查像素是否在当前平面的掩码内（值>0,即255）
                {
                    //只重建属于当前平面的像素
                    //1. 获取像素的 2D 齐次坐标 (y, x, 1) - 匹配 getLabel 中的坐标系
                    Vector3d p_2d_h(static_cast<double>(r),static_cast<double>(c),1.0); //像素坐标[r,c]转为齐次坐标[y,x,1]
                    
                     // 2. 计算从相机中心出发经过该像素的 3D 射线方向 (非归一化即可)
                    Vector3d dir = K_inv * p_2d_h; //dir表示从相机坐标系原点 到像素点在相机坐标系下点 的射线方向
                    
                    //3.射线方程：X=λdir 平面方程：(n^T * X + d = 0 计算射线与平面的交点
                    //λdir 是射线上的一个点，具体位置由 λ确定 dir 本身定义了射线的方向，而 λ确定了沿这个方向的距离。
                    // 将射线带入平面求交点 n^T(λ·dir)+d = 0 λ = -d/n^t·dir 交点X=λdir
                    double n_dot_dir = n.dot(dir);
                     // 检查分母是否接近零 (射线平行于平面，无交点或无穷多交点)
                     if (std::abs(n_dot_dir) < 1e-9) {
                        // std::cerr << "警告: 像素 (" << r << "," << c << ") 的射线平行于平面 " << axis << "，跳过。" << std::endl;
                        continue; // 跳过这个点
                    }

                    //计算投影深度lambda
                    double lambda = -d / n_dot_dir;
                    if (lambda <=0 )
                    {
                       //λ<0表示交点在相机后方(沿射线方向反方向).跳过
                    }

                    //4.计算交点坐标
                    Vector3d p_3d = lambda*dir;

                    //5.存储3D点坐标(相机坐标系到该平面每个像素点在三维空间中的射线与该平面的交点，即恢复该平面)
                    result.points_3d.push_back(p_3d);
                    result.colors.push_back(img.at<cv::Vec3b>(r,c));
                    
                    
                }
                
            }
            
        }
        
    }
    
    

}

/* 8
 * 功能：将重建的点云保存为 PLY (Polygon File Format) 格式文件。
 */
bool createPlyOutput(const ReconstructionResult& result, const std::string& filename)
{
   std::ofstream ply_file(filename); //创建输出文件流
   if (!ply_file.is_open())
   {
     std::cerr << "错误: 无法打开 PLY 文件用于写入: " << filename << std::endl;
     return false;
   }

   //检查点数和颜色是否匹配
   size_t num_vertices = result.points_3d.size();
//    if (num_vertices != result.colors.size())
//    {
//      std::cerr << "错误: 三维点数量 (" << num_vertices << ") 与颜色数量 (" << result.colors.size() << ") 不匹配。" << std::endl;
//      ply_file.close(); // 关闭文件
//      return false;
//    }

   //--写入PLY文件头--
     // ---- 写入 PLY 文件头 ----
    ply_file << "ply\n";                            // 文件标识
    ply_file << "format ascii 1.0\n";             // 文件格式 (ASCII 文本)
    ply_file << "element vertex " << num_vertices << "\n"; // 顶点元素声明及数量
    ply_file << "property float x\n";               // x 坐标属性 (32位浮点数)
    ply_file << "property float y\n";               // y 坐标属性
    ply_file << "property float z\n";               // z 坐标属性
    ply_file << "property uchar red\n";             // 红色通道属性 (8位无符号字符)
    ply_file << "property uchar green\n";           // 绿色通道属性
    ply_file << "property uchar blue\n";            // 蓝色通道属性
    ply_file << "end_header\n";                     // 文件头结束标记

   //---写入顶点数据---
   ply_file << std::fixed <<std::setprecision(6); //设置浮点数输出精度
   for (size_t i = 0; i < num_vertices; ++i) //遍历所有点
   {
     const auto& p =result.points_3d[i]; //获取点i的坐标
     const auto& color_bgr = result.colors[i]; //获取第i个点的颜色

     //写入坐标(x,y,z)
     ply_file << p.x() << " " << p.y() << " " << p.z() <<" ";
    //写入颜色(R,G,B顺序) OpenCV是BGR,PLY通常是RGB
    // 写入颜色 (R, G, B 顺序) - 注意 OpenCV 是 BGR，PLY 通常是 RGB
    ply_file << static_cast<int>(color_bgr[2]) << " "  // Red   (BGR[2])
    << static_cast<int>(color_bgr[1]) << " "  // Green (BGR[1])
    << static_cast<int>(color_bgr[0]) << "\n"; // Blue  (BGR[0])
    // 需要将 uchar 转换为 int 输出，否则可能被当作字符处理

    ply_file.close(); //关闭文件
    std::cout <<"PLY文件写入完成:"<<filename<<std::endl;
   }
   
   

   
}
