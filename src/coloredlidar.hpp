#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>

#define Hmax 2048
#define Wmax 3072
#define H 376
#define W 672

/* 自定义的PointXYZRGBIL（pcl没有PointXYZRGBIL、PointXYZRGBI结构体）*/
struct PointXYZRGBIL
{
    PCL_ADD_POINT4D;
    PCL_ADD_RGB;
    uint32_t label;
    PCL_ADD_INTENSITY;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointXYZRGBIL,
        (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(uint32_t, label, label)(float, intensity, intensity))

// 使用pcl::PointXYZRGBNormal结构体代替自定义的PointXYZRGBI，为了livox_color_mapping建图，自定义结构体不支持很多内置函数
typedef pcl::PointXYZRGBNormal PointTypeRGB;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
//全局变量都能访问，图像回调中写，点云回调中读
cv::Vec3b image_color[H][W];

// 内外参
cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);	   // 畸变向量
cv::Mat intrisic = cv::Mat::eye(3, 3, CV_64F);			   // 内参3*3矩阵
cv::Mat intrisicMat(3, 4, cv::DataType<double>::type);	   // 内参3*4的投影矩阵，最后一列是三个零
cv::Mat extrinsicMat_RT(4, 4, cv::DataType<double>::type); // 外参旋转矩阵3*3和平移向量3*1

void CalibrationData(void)
{

    /*extrinsic
    0.0285186 -0.999539 -0.0104295 0.0438889
    -0.0222481 0.00979648 -0.999704 0.0521314
    0.999346 0.0287422 -0.0219584 -0.0350395
    0 0 0 1*/
    extrinsicMat_RT.at<double>(0, 0) =  0.0285186;
    extrinsicMat_RT.at<double>(0, 1) = -0.999539;
    extrinsicMat_RT.at<double>(0, 2) = -0.0104295;
    extrinsicMat_RT.at<double>(0, 3) =  0.0438889;
    extrinsicMat_RT.at<double>(1, 0) = -0.0222481;
    extrinsicMat_RT.at<double>(1, 1) =  0.00979648;
    extrinsicMat_RT.at<double>(1, 2) = -0.999704;
    extrinsicMat_RT.at<double>(1, 3) =  0.0521314;
    extrinsicMat_RT.at<double>(2, 0) =  0.999346;
    extrinsicMat_RT.at<double>(2, 1) =  0.0287422;
    extrinsicMat_RT.at<double>(2, 2) = -0.0219584;
    extrinsicMat_RT.at<double>(2, 3) = -0.0350395;
    extrinsicMat_RT.at<double>(3, 0) = 0.0;
    extrinsicMat_RT.at<double>(3, 1) = 0.0;
    extrinsicMat_RT.at<double>(3, 2) = 0.0;
    extrinsicMat_RT.at<double>(3, 3) = 1.0;

    /*Intrinsic
    1059.649612174006 0 974.722398656767
    0 1058.75044971954 547.363720882016
    0 0 1
    Distortion
    -0.03932215166914082 0.0086586268606816 -0.000472530659073111 0.000197192778323369 -0.00477887733572879*/

    intrisicMat.at<double>(0, 0) = intrisic.at<double>(0, 0) = 1058.69;
    intrisicMat.at<double>(0, 1) = 0.000000e+00;
    intrisicMat.at<double>(0, 2) = intrisic.at<double>(0, 2) = 975.13;
    intrisicMat.at<double>(0, 3) = 0.000000e+00;
    intrisicMat.at<double>(1, 0) = 0.000000e+00;
    intrisicMat.at<double>(1, 1) = intrisic.at<double>(1, 1) = 1058.14;
    intrisicMat.at<double>(1, 2) = intrisic.at<double>(1, 2) = 549.112;
    intrisicMat.at<double>(1, 3) = 0.000000e+00;
    intrisicMat.at<double>(2, 0) = 0.000000e+00;
    intrisicMat.at<double>(2, 1) = 0.000000e+00;
    intrisicMat.at<double>(2, 2) = 1.000000e+00;
    intrisicMat.at<double>(2, 3) = 0.000000e+00;
    distCoeffs.at<double>(0) = -0.0434855;
    distCoeffs.at<double>(1) =  0.0127718;
    distCoeffs.at<double>(2) = -0.000706572;
    distCoeffs.at<double>(3) = -0.000540132;
    distCoeffs.at<double>(4) = -0.00585071;

}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image; //image_raw就是我们得到的图像了
        // 去畸变，可选
        // cv::Mat map1, map2;
        // cv::Size imageSize = image.size();
        // cv::initUndistortRectifyMap(intrisic, distCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(intrisic, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
        // cv::remap(image, image, map1, map2, cv::INTER_LINEAR); // correct the distortion
        // cv::imwrite("1.bmp",image);
        for (int row = 0; row < H; row++)
        {
            for (int col = 0; col < W; col++)
            {
                image_color[row][col] = (cv::Vec3b)image.at<cv::Vec3b>(row, col);
            }
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not conveextrinsicMat_RT from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}
