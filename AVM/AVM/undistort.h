#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
class Undistort {
 public:
  Undistort();
  //Í¼ÏñµÄÈ¥»û±ä
  cv::Mat undistort_func(cv::Mat img, vector<cv::Mat> &remap_table);
  
 private:
  void getUndistortMap(vector<cv::Mat> &remap_table, int undist_w, int undist_h,
                       cv::Mat intrinsic_undis, cv::Mat intrinsic_fish,
                       cv::Vec4d undis_param, float fish_scale);
  int m_undis_width, m_undis_height;
  float m_fish_scale, m_focal_length, m_dx, m_dy, m_fish_width, m_fish_height,
      m_undis_scale;
  cv::Vec4d m_undis2fish_params;
  cv::Mat m_intrinsic_undis, m_intrinsic;

};
