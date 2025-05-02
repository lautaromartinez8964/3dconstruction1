# ... (cmake_minimum_required, project) ...

# --- 查找依赖库 ---
find_package(Eigen3 REQUIRED)
# 查找 OpenCV 时，确保你指定了需要的组件
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs) # 根据需要添加更多组件

# --- 移除旧的 include_directories ---
# !!! 删除或注释掉下面这几行 (如果存在) !!!
# include_directories(${EIGEN3_INCLUDE_DIR})
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(include) # 可以保留这一行，用于你自己的 include 目录

# --- 定义重建逻辑库 (或直接用于可执行文件) ---
add_library(reconstruction_lib src/reconstruction.cpp)

# !!! 核心改动：使用 target_link_libraries 链接 Target !!!
# 这会将 OpenCV 的 include 目录和库文件正确地关联到你的 target
# PUBLIC 表示链接到 reconstruction_lib 的目标 (如 main.cpp) 也能访问 OpenCV
target_link_libraries(reconstruction_lib PUBLIC
    Eigen3::Eigen         # Eigen 的 CMake Target
    ${OpenCV_LIBS}        # OpenCV 找到的库列表 (或者更现代的 OpenCV::core OpenCV::imgproc 等)
)

# 如果 reconstruction_lib 只是个实现，你可能需要为它添加 include 目录
target_include_directories(reconstruction_lib PUBLIC include) # 让你自己的 include 目录对链接者可见

# --- 定义可执行文件 ---
add_executable(reconstructor src/main.cpp)
# 将可执行文件链接到你的库
target_link_libraries(reconstructor PRIVATE reconstruction_lib)
# 注意：因为 reconstruction_lib 是 PUBLIC 链接 OpenCV，所以 reconstructor 不需要再显式链接 OpenCV
# 如果你没有单独的 reconstruction_lib，直接链接可执行文件：
# target_link_libraries(reconstructor PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
# target_include_directories(reconstructor PRIVATE include) # 添加你自己的 include 目录