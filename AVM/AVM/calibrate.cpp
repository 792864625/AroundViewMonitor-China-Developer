#include "calibrate.h"


// Opencv Source Code
/***************************************************************************************************/
// COMPUTE FAST HISTOGRAM GRADIENT
template <typename ArrayContainer>
static void icvGradientOfHistogram256(const ArrayContainer& piHist,
                                      ArrayContainer& piHistGrad) {
  CV_DbgAssert(piHist.size() == 256);
  CV_DbgAssert(piHistGrad.size() == 256);
  piHistGrad[0] = 0;
  int prev_grad = 0;
  for (int i = 1; i < 255; ++i) {
    int grad = piHist[i - 1] - piHist[i + 1];
    if (std::abs(grad) < 100) {
      if (prev_grad == 0)
        grad = -100;
      else
        grad = prev_grad;
    }
    piHistGrad[i] = grad;
    prev_grad = grad;
  }
  piHistGrad[255] = 0;
}
/***************************************************************************************************/

/***************************************************************************************************/
// SMOOTH HISTOGRAM USING WINDOW OF SIZE 2*iWidth+1
template <int iWidth_, typename ArrayContainer>
static void icvSmoothHistogram256(const ArrayContainer& piHist,
                                  ArrayContainer& piHistSmooth,
                                  int iWidth = 0) {
  CV_DbgAssert(iWidth_ == 0 || (iWidth == iWidth_ || iWidth == 0));
  iWidth = (iWidth_ != 0) ? iWidth_ : iWidth;
  CV_Assert(iWidth > 0);
  CV_DbgAssert(piHist.size() == 256);
  CV_DbgAssert(piHistSmooth.size() == 256);
  for (int i = 0; i < 256; ++i) {
    int iIdx_min = std::max(0, i - iWidth);
    int iIdx_max = std::min(255, i + iWidth);
    int iSmooth = 0;
    for (int iIdx = iIdx_min; iIdx <= iIdx_max; ++iIdx) {
      CV_DbgAssert(iIdx >= 0 && iIdx < 256);
      iSmooth += piHist[iIdx];
    }
    piHistSmooth[i] = iSmooth / (2 * iWidth + 1);
  }
}
/***************************************************************************************************/

/***************************************************************************************************/
// COMPUTE INTENSITY HISTOGRAM OF INPUT IMAGE
template <typename ArrayContainer>
static void icvGetIntensityHistogram256(const cv::Mat& img,
                                        ArrayContainer& piHist) {
  for (int i = 0; i < 256; i++) piHist[i] = 0;
  // sum up all pixel in row direction and divide by number of columns
  for (int j = 0; j < img.rows; ++j) {
    const uchar* row = img.ptr<uchar>(j);
    for (int i = 0; i < img.cols; i++) {
      piHist[row[i]]++;
    }
  }
}
/***************************************************************************************************/

// Opencv source code
static cv::Mat icvBinarizationHistogramBased(cv::Mat img, int fish_undis_flag) {
  CV_Assert(img.channels() == 1 && img.depth() == CV_8U);
  int iCols = img.cols;
  int iRows = img.rows;
  int iMaxPix = iCols * iRows;
  int iMaxPix1 = iMaxPix / 100;
  const int iNumBins = 256;
  const int iMaxPos = 20;
  cv::AutoBuffer<int, 256> piHistIntensity(iNumBins);
  cv::AutoBuffer<int, 256> piHistSmooth(iNumBins);
  cv::AutoBuffer<int, 256> piHistGrad(iNumBins);
  cv::AutoBuffer<int> piMaxPos(iMaxPos);

  icvGetIntensityHistogram256(img, piHistIntensity);

  // first smooth the distribution
  // const int iWidth = 1;
  icvSmoothHistogram256<1>(piHistIntensity, piHistSmooth);

  // compute gradient
  icvGradientOfHistogram256(piHistSmooth, piHistGrad);

  // check for zero points
  unsigned iCntMaxima = 0;
  for (int i = iNumBins - 2; (i > 2) && (iCntMaxima < iMaxPos); --i) {
    if ((piHistGrad[i - 1] < 0) && (piHistGrad[i] > 0)) {
      int iSumAroundMax =
          piHistSmooth[i - 1] + piHistSmooth[i] + piHistSmooth[i + 1];
      if (!(iSumAroundMax < iMaxPix1 && i < 64)) {
        piMaxPos[iCntMaxima++] = i;
      }
    }
  }

  // DPRINTF("HIST: MAXIMA COUNT: %d (%d, %d, %d, ...)", iCntMaxima,
  //         iCntMaxima > 0 ? piMaxPos[0] : -1, iCntMaxima > 1 ? piMaxPos[1] :
  //         -1, iCntMaxima > 2 ? piMaxPos[2] : -1);

  int iThresh = 0;

  CV_Assert((size_t)iCntMaxima <= piMaxPos.size());

  // DPRINTF("HIST: MAXIMA COUNT: %d (%d, %d, %d, ...)", iCntMaxima,
  //         iCntMaxima > 0 ? piMaxPos[0] : -1, iCntMaxima > 1 ? piMaxPos[1] :
  //         -1, iCntMaxima > 2 ? piMaxPos[2] : -1);

  if (iCntMaxima == 0) {
    // no any maxima inside (only 0 and 255 which are not counted above)
    // Does image black-write already?
    const int iMaxPix2 = iMaxPix / 2;
    for (int sum = 0, i = 0; i < 256;
         ++i)  // select mean intensity, median intensity£¿
    {
      sum += piHistIntensity[i];
      if (sum > iMaxPix2) {
        iThresh = i;
        break;
      }
    }
  } else if (iCntMaxima == 1) {
    iThresh = piMaxPos[0] / 2;
  } else if (iCntMaxima == 2) {
    iThresh = (piMaxPos[0] + piMaxPos[1]) / 2;
  } else  // iCntMaxima >= 3
  {
    // CHECKING THRESHOLD FOR WHITE
    int iIdxAccSum = 0, iAccum = 0;
    for (int i = iNumBins - 1; i > 0; --i) {
      iAccum += piHistIntensity[i];
      // iMaxPix/18 is about 5,5%, minimum required number of pixels required,
      // here we modify it to /5 for white part of chessboard
      if (iAccum > (iMaxPix / 5)) {
        iIdxAccSum = i;
        break;
      }
    }

    unsigned iIdxBGMax = 0;
    int iBrightMax = piMaxPos[0];
    // printf("iBrightMax = %d\n", iBrightMax);
    // find the nearest zero point to the white part bar
    for (unsigned n = 0; n < iCntMaxima - 1; ++n) {
      iIdxBGMax = n + 1;
      if (piMaxPos[n] < iIdxAccSum) {
        break;
      }
      iBrightMax = piMaxPos[n];
    }

    // CHECKING THRESHOLD FOR BLACK
    int iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];

    // IF TOO CLOSE TO 255, jump to next maximum
    if (piMaxPos[iIdxBGMax] >= 250 && iIdxBGMax + 1 < iCntMaxima) {
      iIdxBGMax++;
      iMaxVal = piHistIntensity[piMaxPos[iIdxBGMax]];
    }

    // the left max number point's value corresponds to the black threshold
    for (unsigned n = iIdxBGMax + 1; n < iCntMaxima; n++) {
      if (piHistIntensity[piMaxPos[n]] >= iMaxVal) {
        iMaxVal = piHistIntensity[piMaxPos[n]];
        iIdxBGMax = n;
      }
    }

    // SETTING THRESHOLD FOR BINARIZATION

    int iDist2 = (iBrightMax - piMaxPos[iIdxBGMax]) / 2;
    iThresh = iBrightMax - iDist2;

    /*cout << "threshold Max = " << iBrightMax << endl;
    cout << "threshold min = " << piMaxPos[iIdxBGMax] << endl;
    cout << "threshold iThresh = " << iThresh << endl;*/
    // fish dark
    if (fish_undis_flag == 0) {
      auto temp = static_cast<float>(iThresh) * 0.8f;
      iThresh = static_cast<int>(temp);
    }
    // DPRINTF("THRESHOLD SELECTED = %d, BRIGHTMAX = %d, DARKMAX = %d", iThresh,
    //         iBrightMax, piMaxPos[iIdxBGMax]);
  }
  cv::Mat img_thresh(img.rows, img.cols, img.type());
  if (iThresh > 0) {
    img_thresh = (img >= iThresh);
  }
  return img_thresh;
}


cv::Mat calibrate::imgAugForPointDetect(const cv::Mat img, float contrast) {
  cv::Mat mat_float_tmp;

  img.convertTo(mat_float_tmp, 21);
  mat_float_tmp = mat_float_tmp / 255;

  // get the max value
  float maxvalue_ = 0;
  for (int i = 0; i < mat_float_tmp.rows; ++i) {
    float* data = mat_float_tmp.ptr<float>(i);
    for (int j = 0; j < mat_float_tmp.cols; ++j) {
      if (data[j] > maxvalue_) {
        maxvalue_ = data[j];
      }
    }
  }

  mat_float_tmp = mat_float_tmp / maxvalue_;

  pow(mat_float_tmp, contrast, mat_float_tmp);
  mat_float_tmp = mat_float_tmp * 255;

  cv::Mat contrast_img;
  mat_float_tmp.convertTo(contrast_img, 16);

  return contrast_img;
}


void findRectangle(cv::Mat img, std::vector<float> valid_region_y, float max_sz,
                   float fish_scale, std::vector<cv::Point2f>& detect_points,
                   int fish_undis_flag) {
  /*
    img is binary image
  */
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<std::vector<cv::Point>> rectangle;
  float min_y_thresh = valid_region_y[0];
  float max_y_thresh = valid_region_y[1];
  int min_size = static_cast<int>(max_sz * (pow(fish_scale, 2)));
  if (fish_undis_flag == 0) {
    auto temp = static_cast<float>(min_size) * 0.5f;
    min_size = static_cast<int>(temp);
  }
  float approx_level = 10 * fish_scale;

  cv::findContours(img, contours, hierarchy, cv::RETR_CCOMP,
                   cv::CHAIN_APPROX_SIMPLE);

  cv::Rect contour_rect;
  cv::Point pt[4];
  for (int idx = (int)(contours.size() - 1); idx >= 0; --idx) {
    auto contour = contours[idx];
    contour_rect = boundingRect(contour);
    if (contour_rect.area() < min_size) {
      continue;
    }

    std::vector<cv::Point> approx_contour;
    cv::approxPolyDP(contour, approx_contour, approx_level, true);

    if (approx_contour.size() == 4) {
      for (int i = 0; i < 4; ++i) pt[i] = approx_contour[i];

      int x_lable = (pt[0].x + pt[2].x) / 2;
      int y_lable = (pt[0].y + pt[2].y) / 2;
      if (img.at<uchar>(y_lable, x_lable) != 0) {
        continue;
      }

      double p = cv::arcLength(approx_contour, true);
      double area = cv::contourArea(approx_contour, false);

      double d1 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[2]));
      double d2 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[3]));

      // philipg.  Only accept those quadrangles which are more square
      // than rectangular and which are big enough
      double d3 = sqrt(cv::normL2Sqr<double>(pt[0] - pt[1]));
      double d4 = sqrt(cv::normL2Sqr<double>(pt[1] - pt[2]));
      if (!(d3 * 5 > d4 && d4 * 5 > d3 && d3 * d4 < area * 10 &&
            area > min_size && d1 >= 0.1 * p && d2 >= 0.1 * p)) {
        continue;
      }
      float contour_y_average =
          static_cast<float>(approx_contour[0].y + approx_contour[1].y +
                             approx_contour[2].y + approx_contour[3].y) /
          4;
      if (contour_y_average < min_y_thresh ||
          contour_y_average > max_y_thresh) {
        continue;
      }

      // fish x directtion center crop
      //      if (fish_undis_flag == 0) {
      //        float contour_x_average =
      //            static_cast<float>(approx_contour[0].x + approx_contour[1].x
      //            +
      //                               approx_contour[2].x +
      //                               approx_contour[3].x) / 4;
      //
      //        //if (contour_x_average < 950 && contour_x_average > 400) {
      //        //  continue;
      //        //}
      //      }

      detect_points.push_back(approx_contour[0]);
      detect_points.push_back(approx_contour[1]);
      detect_points.push_back(approx_contour[2]);
      detect_points.push_back(approx_contour[3]);

      // circle(img, approx_contour[0], 3, Scalar(0, 255, 0), 30);
      // circle(img, approx_contour[1], 3, Scalar(0, 255, 0), 30);
      // circle(img, approx_contour[2], 3, Scalar(0, 255, 0), 30);
      // circle(img, approx_contour[3], 3, Scalar(0, 255, 0), 30);
      // circle(img, cv::Point(x_lable, y_lable), 1, Scalar(0, 0, 255), 1);
    } else {
      continue;
    }
  }
}
bool cmp(std::vector<cv::Point> A, std::vector<cv::Point> B) {
  return (contourArea(A) > contourArea(B));
}

bool cmpYmax(cv::Point A, cv::Point B) { return (A.y > B.y); }

bool cmpXmax(cv::Point A, cv::Point B) { return (A.x > B.x); }

bool cmpYmin(cv::Point A, cv::Point B) { return (A.y < B.y); }

bool cmpXmin(cv::Point A, cv::Point B) { return (A.x < B.x); }
static std::vector<cv::Point2f> detectPointsSort(
    std::vector<cv::Point2f> points) {
  /*
    return points by row up to down
  */
  sort(points.begin(), points.end(), cmpYmin);
  std::vector<cv::Point2f> mid1(points.begin(), points.begin() + 4);
  std::vector<cv::Point2f> mid2(points.begin() + 4, points.begin() + 8);
  sort(mid1.begin(), mid1.end(), cmpXmin);
  sort(mid2.begin(), mid2.end(), cmpXmin);
  std::vector<cv::Point2f> sortedPoints(mid1.begin(), mid1.end());
  sortedPoints.insert(sortedPoints.end(), mid2.begin(), mid2.end());
  return sortedPoints;
}

void calibrate::solveRtFromPnP(std::vector<cv::Point2f> corners2D,
                    std::vector<cv::Point3f> obj3D, cv::Mat intrinsic,
                    cv::Mat& R, cv::Mat& t) {
  cv::Mat r = cv::Mat::zeros(3, 1, CV_32FC1);
  if (corners2D.size() != obj3D.size()) {
    std::vector<cv::Point2f> corners(corners2D.begin(), corners2D.begin() + 8);
    std::vector<cv::Point3f> obj(obj3D.begin(), obj3D.begin() + 8);
    cv::solvePnP(obj, corners, intrinsic, cv::Mat(), r, t);

  } else {
    cv::solvePnP(obj3D, corners2D, intrinsic, cv::Mat(), r, t);
  }

  Rodrigues(r, R);
}

calibrate::calibrate(){
  float fish_scale = 0.5;
  float focal_length = 910;
  int dx = 3;
  int dy = 3;
  int fish_width = 1280;
  int fish_height = 960;
  float undis_scale = 1.55;
  m_fish2undis_params = {-0.05611147, -0.05377447,
                         0.0115717, 0.0030788};


  m_intrinsic_undis =
      (cv::Mat_<float>(3, 3) << focal_length / dx * fish_scale, 0,
       fish_width / 2 * undis_scale, 0,
       focal_length / dy * fish_scale, fish_height / 2 * undis_scale,
       0, 0, 1);

  m_intrinsic =
      (cv::Mat_<float>(3, 3) << focal_length / dx, 0, fish_width / 2, 0,
       focal_length / dy, fish_height / 2, 0, 0, 1);




  //2D Demo Params
  
  m_spread_w = 520;
  m_spread_h = 960;
  m_block_size = 120;
  m_screen_w = 616;
  m_screen_h = 880;
  m_scale = 1.2857142857;
  m_car_rect_3d_x = 115;
  m_car_rect_3d_y = 260;

  m_total_w = static_cast<int>(m_screen_w * m_scale);  // world coord
  m_total_h = static_cast<int>(m_screen_h * m_scale);
  m_shift_w = (m_total_w - m_spread_w) / 2;
  m_shift_h = (m_total_h - m_spread_h) / 2;

  m_Xl = m_total_w / 2 - m_car_rect_3d_x;
  m_Xr = m_total_w - m_Xl;
  m_Yt = m_total_h / 2 - m_car_rect_3d_y;
  m_Yb = m_total_h - m_Yt;


  m_Xl_screen = static_cast<int>(m_Xl / m_scale);
  m_Yt_screen = static_cast<int>(m_Yt / m_scale);
  m_Xr_screen = static_cast<int>(ceil(m_Xr / m_scale));
  m_Yb_screen = static_cast<int>(ceil(m_Yb / m_scale));



  m_corner_bird_front = std::vector<cv::Point2i>(8);
  m_corner_bird_back = std::vector<cv::Point2i>(8);
  m_corner_bird_left = std::vector<cv::Point2i>(8);
  m_corner_bird_right = std::vector<cv::Point2i>(8);

  /*******************************************************front
   * back**************************************************/
  m_corner_bird_front[0].x = m_shift_w;
  m_corner_bird_front[1].x = m_shift_w + m_block_size;
  m_corner_bird_front[2].x = m_total_w - m_shift_w - m_block_size;
  m_corner_bird_front[3].x = m_total_w - m_shift_w;
  m_corner_bird_front[4].x = m_shift_w;
  m_corner_bird_front[5].x = m_shift_w + m_block_size;
  m_corner_bird_front[6].x = m_total_w - m_shift_w - m_block_size;
  m_corner_bird_front[7].x = m_total_w - m_shift_w;

  m_corner_bird_front[0].y = m_shift_h;
  m_corner_bird_front[1].y = m_shift_h;
  m_corner_bird_front[2].y = m_shift_h;
  m_corner_bird_front[3].y = m_shift_h;
  m_corner_bird_front[4].y = m_shift_h + m_block_size;
  m_corner_bird_front[5].y = m_shift_h + m_block_size;
  m_corner_bird_front[6].y = m_shift_h + m_block_size;
  m_corner_bird_front[7].y = m_shift_h + m_block_size;

  
  m_corner_bird_back = m_corner_bird_front;

  /*******************************************************left**************************************************/
  m_corner_bird_left[0].x = m_shift_h;
  m_corner_bird_left[1].x = m_shift_h + m_block_size;
  m_corner_bird_left[2].x = m_total_h - m_shift_h - m_block_size;
  m_corner_bird_left[3].x = m_total_h - m_shift_h;
  m_corner_bird_left[4].x = m_shift_h;
  m_corner_bird_left[5].x = m_shift_h + m_block_size;
  m_corner_bird_left[6].x = m_total_h - m_shift_h - m_block_size;
  m_corner_bird_left[7].x = m_total_h - m_shift_h;

  m_corner_bird_left[0].y = m_shift_w;
  m_corner_bird_left[1].y = m_shift_w;
  m_corner_bird_left[2].y = m_shift_w;
  m_corner_bird_left[3].y = m_shift_w;
  m_corner_bird_left[4].y = m_shift_w + m_block_size;
  m_corner_bird_left[5].y = m_shift_w + m_block_size;
  m_corner_bird_left[6].y = m_shift_w + m_block_size;
  m_corner_bird_left[7].y = m_shift_w + m_block_size;

  /*******************************************************right**************************************************/
  m_corner_bird_right[0].x = m_shift_h;
  m_corner_bird_right[1].x = m_shift_h + m_block_size;
  m_corner_bird_right[2].x = m_total_h - m_shift_h - m_block_size;
  m_corner_bird_right[3].x = m_total_h - m_shift_h;
  m_corner_bird_right[4].x = m_shift_h;
  m_corner_bird_right[5].x = m_shift_h + m_block_size;
  m_corner_bird_right[6].x = m_total_h - m_shift_h - m_block_size;
  m_corner_bird_right[7].x = m_total_h - m_shift_h;

  m_corner_bird_right[0].y = m_shift_w;
  m_corner_bird_right[1].y = m_shift_w;
  m_corner_bird_right[2].y = m_shift_w;
  m_corner_bird_right[3].y = m_shift_w;
  m_corner_bird_right[4].y = m_shift_w + m_block_size;
  m_corner_bird_right[5].y = m_shift_w + m_block_size;
  m_corner_bird_right[6].y = m_shift_w + m_block_size;
  m_corner_bird_right[7].y = m_shift_w + m_block_size;

  cv::Mat msk1 = cv::imread("maskFront.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat msk2 = cv::imread("maskBack.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat msk3 = cv::imread("maskLeft.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat msk4 = cv::imread("maskRight.jpg", cv::IMREAD_GRAYSCALE);
  msk1.convertTo(msk1, CV_32F, 1 / 255.0f);
  msk2.convertTo(msk2, CV_32F, 1 / 255.0f);
  msk3.convertTo(msk3, CV_32F, 1 / 255.0f);
  msk4.convertTo(msk4, CV_32F, 1 / 255.0f);
  m_bird_mask = {msk1, msk2, msk3, msk4};

}

void warpPointInverse(cv::Vec2f& warp_xy, float map_center_h,
                                     float map_center_w, float x_, float y_,
                                     float scale) {
  warp_xy[0] = x_ * scale + map_center_w;
  warp_xy[1] = y_ * scale + map_center_h;
}
cv::Vec2f warpFisheye2Undist(
    float fish_scale, float f_dx, float f_dy, float undis_center_h,
    float undis_center_w, float fish_center_h, float fish_center_w,
    cv::Vec4d undis_param, float x, float y) {
  // f_dx *= fish_scale;
  // f_dy *= fish_scale;
  float y_ = (y - fish_center_h) / f_dy;  // normalized plane
  float x_ = (x - fish_center_w) / f_dx;
  float r_distorted = static_cast<float>(sqrt(pow(x_, 2) + pow(y_, 2)));

  float r_distorted_p2 = r_distorted * r_distorted;
  float r_distorted_p3 = r_distorted_p2 * r_distorted;
  float r_distorted_p4 = r_distorted_p2 * r_distorted_p2;
  float r_distorted_p5 = r_distorted_p2 * r_distorted_p3;
  float angle_undistorted = static_cast<float>(
      r_distorted + undis_param[0] * r_distorted_p2 +
      undis_param[1] * r_distorted_p3 + undis_param[2] * r_distorted_p4 +
      undis_param[3] * r_distorted_p5);
  // scale
  float r_undistorted = tanf(angle_undistorted);

  float scale =
      r_undistorted /
      (r_distorted + 0.00001f);  // scale = r_dis on the camera img plane
                                 // divide r_undis on the normalized plane
  cv::Vec2f warp_xy;

  float xx = (x - fish_center_w) * fish_scale;
  float yy = (y - fish_center_h) * fish_scale;

  warpPointInverse(warp_xy, undis_center_h, undis_center_w, xx, yy, scale);

  return warp_xy;
}


bool calibrate::detectPoints(cv::Mat img, float max_sz, float fish_scale,
                             std::vector<cv::Point2f>& detect_points,
                             int fish_undis_flag) {
  float max_y_thresh = static_cast<float>(0.7f * img.rows);
  float min_y_thresh = static_cast<float>(0.2f * img.rows);
  std::vector<float> y_valid_area{min_y_thresh, max_y_thresh};

  if (img.channels() != 1) {
    cv::cvtColor(img, img, COLOR_BGR2GRAY);
  }

  // ==================== image preprocess =====================
  float contrast = 2.5f;
  cv::Mat img_contrast = img;
  img_contrast = imgAugForPointDetect(img, contrast);
  // ===========================================================

  cv::Mat img_thresh =
      icvBinarizationHistogramBased(img_contrast, fish_undis_flag);

  findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points,
                fish_undis_flag);

  if (detect_points.size() != 8) {
    // planB
    cv::adaptiveThreshold(img_contrast, img_thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          401, 5);
    detect_points.clear();
    findRectangle(img_thresh, y_valid_area, max_sz, fish_scale, detect_points,
                  fish_undis_flag);
  }

  //draw points for debug
   for (int i = 0; i < detect_points.size(); i++) {
     cv::circle(img_thresh, detect_points[i], 1, cv::Scalar(0, 255, 0), 5);
   }


  if (detect_points.size() != 8) {
    return false;
  }
  cv::cornerSubPix(
      img, detect_points, cv::Size(9, 9), cv::Size(-1, -1),
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

  detect_points = detectPointsSort(detect_points);

  float f_dx = m_intrinsic.at<float>(0, 0);
  float f_dy = m_intrinsic.at<float>(1, 1);
  float fish_center_x = m_intrinsic.at<float>(0, 2);
  float fish_center_y = m_intrinsic.at<float>(1, 2);
  float undis_center_x = m_intrinsic_undis.at<float>(0, 2);
  float undis_center_y = m_intrinsic_undis.at<float>(1, 2);
  cv::Vec4d fish2undis_params = m_fish2undis_params;
  cv::Vec2f xy;

  std::vector<std::string> view_direction{"front", "back", "left", "right"};
  std::vector<std::vector<cv::Point2f>> undis_pts_opencv(4);
  for (int j = 0; j < 8; j++) {
    xy = warpFisheye2Undist(fish_scale, f_dx, f_dy, undis_center_y,
                            undis_center_x, fish_center_y, fish_center_x,
                            fish2undis_params, detect_points[j].x,
                            detect_points[j].y);

    detect_points[j].x = xy[0];
    detect_points[j].y = xy[1];
  }
  return true;
}


void calibrate::calib(vector<cv::Mat> fish) {

  //img undistort mapping Lut
  Undistort undistort_handle = Undistort(); 
  std::vector<cv::Mat> undis2dis_front = std::vector<cv::Mat>();
  std::vector<cv::Mat> undis2dis_back = std::vector<cv::Mat>();
  std::vector<cv::Mat> undis2dis_left = std::vector<cv::Mat>();
  std::vector<cv::Mat> undis2dis_right = std::vector<cv::Mat>();


  cv::Mat front_undis =
      undistort_handle.undistort_func(fish[0], undis2dis_front);
  cv::Mat back_undis =
      undistort_handle.undistort_func(fish[1], undis2dis_back);
  cv::Mat left_undis =
      undistort_handle.undistort_func(fish[2], undis2dis_left);
  cv::Mat right_undis =
      undistort_handle.undistort_func(fish[3], undis2dis_right);
  std::vector<cv::Mat> undisImgSet = {front_undis, back_undis, left_undis,
                                      right_undis};

  std::vector<std::vector<cv::Mat>> undis2dis_map = {
      undis2dis_front, undis2dis_back, undis2dis_left, undis2dis_right};

  //undis pts
  detectPoints(fish[0], 20000, 0.5, m_corner_front, 0);
  detectPoints(fish[1], 20000, 0.5, m_corner_back, 0);
  detectPoints(fish[2], 20000, 0.5, m_corner_left, 0);
  detectPoints(fish[3], 20000, 0.5, m_corner_right, 0);
  /*for (int j = 0; j < 8; j++) {
    circle(front_undis, m_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
    circle(front_undis, m_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
    circle(front_undis, m_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
    circle(front_undis, m_corner_front[j], 3, cv::Scalar(0, 255, 0), 1);
  }*/
  



  //2D Demo
  //
  

  std::vector<cv::Mat> birdSet(4);
  projRectPoints(undisImgSet, birdSet, m_Homo_RansacF, m_Homo_RansacB,
                 m_Homo_RansacL, m_Homo_RansacR);

  getWarpLutOpencv(undis2dis_map);
  cv::remap(fish[0], birdSet[0], m_lut[0][0], m_lut[0][1], cv::INTER_LINEAR);
  cv::remap(fish[1], birdSet[1], m_lut[1][0], m_lut[1][1], cv::INTER_LINEAR);
  cv::remap(fish[2], birdSet[2], m_lut[2][0], m_lut[2][1], cv::INTER_LINEAR);
  cv::remap(fish[3], birdSet[3], m_lut[3][0], m_lut[3][1], cv::INTER_LINEAR);

  cv::imwrite("birt_front.jpg", birdSet[0]);
  cv::imwrite("birt_back.jpg", birdSet[1]);
  cv::imwrite("birt_left.jpg", birdSet[2]);
  cv::imwrite("birt_right.jpg", birdSet[3]);

  cv::Mat output2D = combineBlend(birdSet);
  cv::imwrite("bev.jpg", output2D);
}



/***************************2D Demo********************************/

cv::Mat calibrate::combineBlend(std::vector<cv::Mat>& birdSet) {
  cv::Mat output(m_screen_h, m_screen_w, CV_8UC3, cv::Scalar(0, 0, 0));
  for (size_t i = 0; i < birdSet.size(); i++) {
    birdSet[i].convertTo(birdSet[i], CV_32FC3);
  }
  // front/back
  for (int i = 0; i < m_bird_mask[0].rows; i++) {
    for (int j = 0; j < m_bird_mask[0].cols; j++) {
      for (int channel = 0; channel < 3; channel++) {
        output.at<cv::Vec3b>(i, j)[channel] =
            static_cast<uchar>(birdSet[0].at<cv::Vec3f>(i, j)[channel] *
                               m_bird_mask[0].at<float>(i, j));

        output.at<cv::Vec3b>(i + m_Yb_screen, j)[channel] =
            static_cast<uchar>(birdSet[1].at<cv::Vec3f>(i, j)[channel] *
                               m_bird_mask[1].at<float>(i, j));
      }
    }
  }
  // left right
  for (int i = 0; i < m_bird_mask[2].rows; i++) {
    for (int j = 0; j < m_bird_mask[2].cols; j++) {
      for (int channel = 0; channel < 3; channel++) {
        output.at<cv::Vec3b>(i, j)[channel] +=
            static_cast<uchar>(birdSet[2].at<cv::Vec3f>(i, j)[channel] *
                               m_bird_mask[2].at<float>(i, j));

        output.at<cv::Vec3b>(i, j + m_Xr_screen)[channel] +=
            static_cast<uchar>(birdSet[3].at<cv::Vec3f>(i, j)[channel] *
                               m_bird_mask[3].at<float>(i, j));
      }
    }
  }
  return output;
}

void calibrate::projRectPoints(const std::vector<cv::Mat>& undisImgSet,
                                 std::vector<cv::Mat>& birdSet,
                                 cv::Mat& m_Homo_RansacF, cv::Mat& m_Homo_RansacB,
                                 cv::Mat& m_Homo_RansacL, cv::Mat& m_Homo_RansacR) {
  //
  m_Homo_RansacF = cv::findHomography(m_corner_front, m_corner_bird_front, 0);
  m_Homo_RansacB = cv::findHomography(m_corner_back, m_corner_bird_back, 0);
  m_Homo_RansacL = cv::findHomography(m_corner_left, m_corner_bird_left, 0);
  m_Homo_RansacR = cv::findHomography(m_corner_right, m_corner_bird_right, 0);

  cv::warpPerspective(undisImgSet[0], birdSet[0], m_Homo_RansacF,
                      cv::Size(m_total_w, m_Yt), cv::INTER_LINEAR);
  cv::warpPerspective(undisImgSet[1], birdSet[1], m_Homo_RansacB,
                      cv::Size(m_total_w, m_Yt), cv::INTER_LINEAR);
  cv::warpPerspective(undisImgSet[2], birdSet[2], m_Homo_RansacL,
                      cv::Size(m_total_h, m_Xl), cv::INTER_LINEAR);
  cv::warpPerspective(undisImgSet[3], birdSet[3], m_Homo_RansacR,
                      cv::Size(m_total_h, m_Xl), cv::INTER_LINEAR);
  
  cv::imwrite("birt_front.jpg", birdSet[0]);
  cv::imwrite("birt_back.jpg", birdSet[1]);
  cv::imwrite("birt_left.jpg", birdSet[2]);
  cv::imwrite("birt_right.jpg", birdSet[3]);
    

  m_Homo_RansacF.convertTo(m_Homo_RansacF, CV_32F);
  m_Homo_RansacB.convertTo(m_Homo_RansacB, CV_32F);
  m_Homo_RansacL.convertTo(m_Homo_RansacL, CV_32F);
  m_Homo_RansacR.convertTo(m_Homo_RansacR, CV_32F);
}


//generate  fish2around Lut
void calibrate::getWarpLutOpencv(
    const std::vector<std::vector<cv::Mat>>& map_set) {
  std::vector<cv::Mat> map_F(2);
  std::vector<cv::Mat> map_B(2);
  std::vector<cv::Mat> map_L(2);
  std::vector<cv::Mat> map_R(2);

  // four directions rotation matrics
  cv::Mat rota_matrixF = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  cv::Mat rota_matrixB =
      (cv::Mat_<float>(3, 3) << -1, 0, m_total_w, 0, -1, m_Yt, 0, 0, 1);
  cv::Mat rota_matrixL =
      (cv::Mat_<float>(3, 3) << 0, -1, m_total_h, 1, 0, 0, 0, 0, 1);
  cv::Mat rota_matrixR =
      (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, m_Xl, 0, 0, 1);

  warpDistort2Around(map_set[0][0], map_set[0][1], 0, m_Yt, 0, m_total_w,
                     rota_matrixF, m_Homo_RansacF, map_F[0], map_F[1], m_Yt,
                     m_total_w);
  warpDistort2Around(map_set[1][0], map_set[1][1], 0, m_Yt, 0, m_total_w,
                     rota_matrixB, m_Homo_RansacB, map_B[0], map_B[1], m_Yt,
                     m_total_w);
  warpDistort2Around(map_set[2][0], map_set[2][1], 0, m_total_h, 0, m_Xl,
                     rota_matrixL, m_Homo_RansacL, map_L[0], map_L[1],
                     m_total_h, m_Xl);
  warpDistort2Around(map_set[3][0], map_set[3][1], 0, m_total_h, 0, m_Xl,
                     rota_matrixR, m_Homo_RansacR, map_R[0], map_R[1],
                     m_total_h, m_Xl);

  m_lut.push_back(map_F);
  m_lut.push_back(map_B);
  m_lut.push_back(map_L);
  m_lut.push_back(map_R);

   // map resize -> Screen Size
  cv::resize(m_lut[0][0], m_lut[0][0], cv::Size(m_screen_w, m_Yt_screen));
  cv::resize(m_lut[0][1], m_lut[0][1], cv::Size(m_screen_w, m_Yt_screen));
  cv::resize(m_lut[1][0], m_lut[1][0], cv::Size(m_screen_w, m_Yt_screen));
  cv::resize(m_lut[1][1], m_lut[1][1], cv::Size(m_screen_w, m_Yt_screen));
  cv::resize(m_lut[2][0], m_lut[2][0], cv::Size(m_Xl_screen, m_screen_h));
  cv::resize(m_lut[2][1], m_lut[2][1], cv::Size(m_Xl_screen, m_screen_h));
  cv::resize(m_lut[3][0], m_lut[3][0], cv::Size(m_Xl_screen, m_screen_h));
  cv::resize(m_lut[3][1], m_lut[3][1], cv::Size(m_Xl_screen, m_screen_h));

}

void matrix_mul_3x3(cv::Mat H, int j, int i, cv::Vec2f& label) {
  float div = H.at<float>(2, 0) * j + H.at<float>(2, 1) * i + H.at<float>(2, 2);
  label[0] =
      (H.at<float>(0, 0) * j + H.at<float>(0, 1) * i + H.at<float>(0, 2)) / div;
  label[1] =
      (H.at<float>(1, 0) * j + H.at<float>(1, 1) * i + H.at<float>(1, 2)) / div;
}
void calibrate::warpDistort2Around(
    const cv::Mat& map1_x, const cv::Mat& map1_y, const int label1,
    const int label2, const int label3, const int label4, const cv::Mat& matrix,
    const cv::Mat& Homo, cv::Mat& mx, cv::Mat& my, const int map_height,
    const int map_width) {
  cv::Mat map2_xR(map_height, map_width, CV_32F);
  cv::Mat map2_yR(map_height, map_width, CV_32F);
  cv::Mat Homo_inverse;
  cv::invert(Homo, Homo_inverse, cv::DECOMP_LU);

  cv::Mat H_matrix = Homo_inverse * matrix;  // H + rotation
  cv::Vec2f label;
  for (int i = label1; i < label2; i++) {
    float* data_x = map2_xR.ptr<float>(i);
    float* data_y = map2_yR.ptr<float>(i);
    for (int j = label3; j < label4; j++) {
      matrix_mul_3x3(H_matrix, j, i, label);
      data_x[j] = label[0];
      data_y[j] = label[1];
    }
  }
  // merge the two map
  remap(map1_y, my, map2_xR, map2_yR, cv::INTER_LINEAR);
  remap(map1_x, mx, map2_xR, map2_yR, cv::INTER_LINEAR);
}
