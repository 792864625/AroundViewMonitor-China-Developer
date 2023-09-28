#include "undistort.h"

void main() { 
	Undistort undistort_handle = Undistort(); 

	
	cv::Mat img = cv::imread("front.png");
        undistort_handle.remap(img);

}