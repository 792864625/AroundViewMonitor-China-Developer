#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include "undistort.h"
using namespace cv;
using namespace std;
class calibrate {
 public:
  calibrate();
  void calib(vector<cv::Mat> fish);
  bool detectPoints(cv::Mat img, float max_sz, float fish_scale,
                    std::vector<cv::Point2f> &detect_points,
                    int fish_undis_flag);
  cv::Mat imgAugForPointDetect(const cv::Mat img, float contrast);
 private: 
     void transFisheye2Undist(std::vector<std::vector<cv::Point2f>>& corners);
     void solveRtFromPnP(std::vector<cv::Point2f> corners2D,
                      std::vector<cv::Point3f> obj3D, cv::Mat intrinsic,
                      cv::Mat& R, cv::Mat& t);
     
  std::vector<cv::Point3f> obj3D_f;
  int a = 0;

  cv::Mat m_intrinsic_undis, m_intrinsic;
  cv::Vec4d m_fish2undis_params;

  std::vector<cv::Point2f> m_corner_front, m_corner_back, m_corner_left,
      m_corner_right;

  //2D Demo
  void projRectPoints(const std::vector<cv::Mat>& undisImgSet,
                      std::vector<cv::Mat>& birdSet, cv::Mat& m_Homo_RansacF,
                      cv::Mat& m_Homo_RansacB, cv::Mat& m_Homo_RansacL,
                      cv::Mat& m_Homo_RansacR);
  void getWarpLutOpencv(const std::vector<std::vector<cv::Mat>>& map_set);
  void warpDistort2Around(const cv::Mat& map1_x, const cv::Mat& map1_y,
                          const int label1, const int label2, const int label3,
                          const int label4, const cv::Mat& matrix,
                          const cv::Mat& Homo, cv::Mat& mx, cv::Mat& my,
                          const int map_height, const int map_width);
  cv::Mat combineBlend(std::vector<cv::Mat>& birdSet);

  std::vector<cv::Point2i> m_corner_bird_front;
  std::vector<cv::Point2i> m_corner_bird_back;
  std::vector<cv::Point2i> m_corner_bird_left;
  std::vector<cv::Point2i> m_corner_bird_right;

  

  int m_spread_w, m_spread_h, m_screen_w, m_screen_h, m_block_size, m_car_rect_3d_x, m_car_rect_3d_y;
  float m_scale;


  int m_shift_w, m_shift_h, m_total_w, m_total_h;
  int m_Xl;
  int m_Xr;
  int m_Yt;
  int m_Yb;
  int m_Xl_screen;
  int m_Xr_screen;
  int m_Yt_screen;
  int m_Yb_screen;

  cv::Mat m_Homo_RansacF;
  cv::Mat m_Homo_RansacB;
  cv::Mat m_Homo_RansacL;
  cv::Mat m_Homo_RansacR;
  std::vector<std::vector<cv::Mat>> m_lut;
  std::vector<cv::Mat> m_bird_mask;

  
};
