#include "undistort.h"


void warpPointOpencv(cv::Vec2f &warp_xy, float map_center_h, float map_center_w,
                     float x_, float y_, float scale) {
  warp_xy[0] = x_ * scale + map_center_w;
  warp_xy[1] = y_ * scale + map_center_h;
}

/*
func: warp from distort to undistort

@param   f_dx:f/dx
@param   f_dy:f/dy
@param   large_center_h:    undis image center y
@param   large_center_w:    undis image center x
@param   fish_center_h:     fish image center y
@param   fish_center_w:     fish image center x
@param   undis_param:       factory param
@param   x:                 input coordinate x on the undis image
@param   y:                 input coordinate y on the undis image
*/
cv::Vec2f warpUndist2Fisheye(float fish_scale, float f_dx, float f_dy,
                             float large_center_h, float large_center_w,
                             float fish_center_h, float fish_center_w,
                             const cv::Vec4d &undis_param, float x, float y) {
  f_dx *= fish_scale;
  f_dy *= fish_scale;
  float y_ = (y - large_center_h) / f_dy;  // normalized plane
  float x_ = (x - large_center_w) / f_dx;
  float r_ = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

  // Look up table
  /*int num = atan(r_) / atan(m_d) * 1024;
  float angle_distorted = m_Lut[num];*/

  float angle_undistorted = atan(r_);  // theta
  float angle_undistorted_p2 = angle_undistorted * angle_undistorted;
  float angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted;
  float angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3;
  float angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5;
  float angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7;

  float angle_distorted = static_cast<float>(
      angle_undistorted + undis_param[0] * angle_undistorted_p3 +
      undis_param[1] * angle_undistorted_p5 +
      undis_param[2] * angle_undistorted_p7 +
      undis_param[3] * angle_undistorted_p9);
  // scale
  float scale = angle_distorted /
                (r_ + 0.00001f);  // scale = r_dis on the camera img plane
                                  // divide r_undis on the normalized plane
  cv::Vec2f warp_xy;

  float xx = (x - large_center_w) / fish_scale;
  float yy = (y - large_center_h) / fish_scale;

  warpPointOpencv(warp_xy, fish_center_h, fish_center_w, xx, yy, scale);

  return warp_xy;
}
void Undistort::getUndistortMap(vector<cv::Mat> &remap_table, int undist_w,
                                int undist_h, cv::Mat intrinsic_undis,
                                cv::Mat intrinsic_fish, cv::Vec4d undis_param,
                                float fish_scale) {
  float fisheye_width = intrinsic_fish.at<float>(0, 2) * 2.0f;
  float fisheye_height = intrinsic_fish.at<float>(1, 2) * 2.0f;
  cv::Mat map_x(undist_h, undist_w, CV_32F);
  cv::Mat map_y(undist_h, undist_w, CV_32F);
  float step_size = 1.0f;

  cv::Mat camera_points;
  for (int i = 0; i < undist_h; i++) {
    float *row_x = map_x.ptr<float>(i);
    float *row_y = map_y.ptr<float>(i);
    for (int j = 0; j < undist_w; j++) {
      cv::Vec2f xy = warpUndist2Fisheye(
          fish_scale, intrinsic_fish.at<float>(0, 0),
          intrinsic_fish.at<float>(1, 1), intrinsic_undis.at<float>(1, 2),
          intrinsic_undis.at<float>(0, 2), intrinsic_fish.at<float>(1, 2),
          intrinsic_fish.at<float>(0, 2), undis_param, static_cast<float>(j),
          static_cast<float>(i));
      // bounds protection
      xy[0] = xy[0] >= 0 ? xy[0] : 0.0f;
      xy[1] = xy[1] >= 0 ? xy[1] : 0.0f;
      xy[0] = xy[0] < fisheye_width ? xy[0] : fisheye_width - 1.0f;
      xy[1] = xy[1] < fisheye_height ? xy[1] : fisheye_height - 1.0f;

      row_x[j] = xy[0];
      row_y[j] = xy[1];
    }
  }
  remap_table.push_back(map_x);
  remap_table.push_back(map_y);
}

void Undistort::remap(cv::Mat img) {
  std::vector<cv::Mat> remap_table = std::vector<cv::Mat>();

  int undis_width = 1984;
  int undis_height = 1488;
  float m_fish_scale = 0.5;

  float m_focal_length = 910;
  float m_dx = 3;
  float m_dy = 3;
  float fish_width = 1280;
  float fish_height = 960;
  float m_undis_scale = 1.5;
  // calib init
  Mat m_intrinsic_undis =
      (cv::Mat_<float>(3, 3) << m_focal_length / m_dx * m_fish_scale, 0,
       fish_width / 2 * m_undis_scale, 0, m_focal_length / m_dy * m_fish_scale,
       fish_height / 2 * m_undis_scale, 0, 0, 1);

  Mat m_intrinsic =
      (cv::Mat_<float>(3, 3) << m_focal_length / m_dx, 0, fish_width / 2, 0,
       m_focal_length / m_dy, fish_height / 2, 0, 0, 1);
  cv::Vec4d m_undis2fish_params = {0.18238692, -0.08579553, 0.03366532,
                                   -0.00561911};
  getUndistortMap(remap_table, undis_width, undis_height, m_intrinsic_undis,
                m_intrinsic, m_undis2fish_params, m_fish_scale);

  cv::Mat undis2dis_mapx = remap_table[0];
  cv::Mat undis2dis_mapy = remap_table[1];

  cv::Mat undis_img;
  cv::remap(img, undis_img, undis2dis_mapx, undis2dis_mapy,
            cv::INTER_LINEAR);

}