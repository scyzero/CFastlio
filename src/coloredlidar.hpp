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
#define H 1080
#define W 1920

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
    -0.248743  -0.967485  -0.0458282  0.6701
    -0.357534  0.135691  -0.92399  0.891248
     0.900164  -0.213451  -0.379661  0.41885
     0  0  0  1*/
    extrinsicMat_RT.at<double>(0, 0) = -0.0120078;
    extrinsicMat_RT.at<double>(0, 1) = -0.999918;
    extrinsicMat_RT.at<double>(0, 2) =  0.00452853;
    extrinsicMat_RT.at<double>(0, 3) =  0.0653615;
    extrinsicMat_RT.at<double>(1, 0) = -0.0804771;
    extrinsicMat_RT.at<double>(1, 1) = -0.00354775;
    extrinsicMat_RT.at<double>(1, 2) = -0.99675;
    extrinsicMat_RT.at<double>(1, 3) =  0.135746;
    extrinsicMat_RT.at<double>(2, 0) =  0.996684;
    extrinsicMat_RT.at<double>(2, 1) = -0.0123332;
    extrinsicMat_RT.at<double>(2, 2) = -0.0804278;
    extrinsicMat_RT.at<double>(2, 3) = 0.0385729;
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

class livox_lidar_color
{
public:
    ros::NodeHandle n;
    sensor_msgs::PointCloud2 msg;																									   //接收到的点云消息
    sensor_msgs::PointCloud2 fusion_msg;																							   //等待发送的点云消息
    ros::Subscriber subCloud = n.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1, &livox_lidar_color::pointCloudCallback, this); //接收点云数据，进入回调函数pointCloudCallback
    ros::Publisher pubCloud = n.advertise<sensor_msgs::PointCloud2>("/livox/color_lidar", 1);										   //建立了一个发布器，方便之后发布加入颜色之后的点云

private:
    //点云回调函数
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>); //livox点云消息包含xyz和intensity
        pcl::fromROSMsg(*laserCloudMsg, *raw_pcl_ptr);										   //把msg消息指针转化为PCL点云
        cv::Mat X(4, 1, cv::DataType<double>::type);
        cv::Mat Y(3, 1, cv::DataType<double>::type);

        pcl::PointCloud<PointTypeRGB>::Ptr fusion_pcl_ptr(new pcl::PointCloud<PointTypeRGB>); //放在这里是因为，每次都需要重新初始化

        for (int i = 0; i < raw_pcl_ptr->points.size(); i++)
        {
            X.at<double>(0, 0) = raw_pcl_ptr->points[i].x;
            X.at<double>(1, 0) = raw_pcl_ptr->points[i].y;
            X.at<double>(2, 0) = raw_pcl_ptr->points[i].z;
            X.at<double>(3, 0) = 1;
            Y = intrisicMat * extrinsicMat_RT * X; //雷达坐标转换到相机坐标，相机坐标投影到像素坐标
            cv::Point pt;						   // (x,y) 像素坐标
            // Y是3*1向量，pt.x是Y的第一个值除以第三个值，pt.y是Y的第二个值除以第三个值，为什么是下面这种写法？？
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
            // std::cout<<Y<<pt<<std::endl;
            if (pt.x >= 0 && pt.x < W && pt.y >= 0 && pt.y < H && raw_pcl_ptr->points[i].x > 0) //&& raw_pcl_ptr->points[i].x>0去掉图像后方的点云
            {

                PointTypeRGB p;
                p.x = raw_pcl_ptr->points[i].x;
                p.y = raw_pcl_ptr->points[i].y;
                p.z = raw_pcl_ptr->points[i].z;
                //点云颜色由图像上对应点确定
                p.b = image_color[pt.y][pt.x][0];
                p.g = image_color[pt.y][pt.x][1];
                p.r = image_color[pt.y][pt.x][2];
                fusion_pcl_ptr->points.push_back(p);


            }
        }

        fusion_pcl_ptr->width = fusion_pcl_ptr->points.size();
        fusion_pcl_ptr->height = 1;
        // std::cout<<  fusion_pcl_ptr->points.size() << std::endl;
        pcl::toROSMsg(*fusion_pcl_ptr, fusion_msg);			   //将点云转化为消息才能发布
        fusion_msg.header.frame_id = "livox_frame";			   //帧id改成和/livox/lidar一样的，同一坐标系
        fusion_msg.header.stamp = laserCloudMsg->header.stamp; // 时间戳和/livox/lidar 一致
        pubCloud.publish(fusion_msg);						   //发布调整之后的点云数据
    }
};

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