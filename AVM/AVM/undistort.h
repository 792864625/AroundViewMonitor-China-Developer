#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
class Undistort {
 public:
  //Í¼ÏñµÄÈ¥»û±ä
  void remap(cv::Mat img);
  void getUndistortMap(
      vector<cv::Mat> &remap_table, int undist_w,
      int undist_h, cv::Mat intrinsic_undis, cv::Mat intrinsic_fish,
      cv::Vec4d undis_param, float fish_scale);
 private:
  void meshGrid(const int w, const int h, const float step_size,
                           cv::Mat &map_x, cv::Mat &map_y);
};
