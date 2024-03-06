#include "undistort.h"
#include "calibrate.h"

/********test undistort*********/
//void main() { 
//	Undistort undistort_handle = Undistort(); 
//
//	
//	cv::Mat img = cv::imread("front.png");
//    undistort_handle.undistort_func(img);
//
//}




/********test calibrate*********/
void main() {
  calibrate calibrate_handle = calibrate();

  cv::Mat front = cv::imread("front.png");
  cv::Mat back = cv::imread("back.png");
  cv::Mat left = cv::imread("left.png");
  cv::Mat right = cv::imread("right.png");
  vector<Mat> fish = {front, back, left, right};
  std::vector<cv::Point2f> corners;
  calibrate_handle.calib(fish);


  return;
}