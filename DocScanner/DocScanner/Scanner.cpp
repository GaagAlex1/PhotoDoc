#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

Mat prepareImage(Mat img) {
	Mat orig, gaus, gray, canny, dil, warp;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gaus, Size(3, 3), 3, 0);
	Canny(gaus, canny, 50, 100);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));

	dilate(canny, dil, kernel);
	return dil;
}

void getSortedAreas(vector<vector<Point>>& contours) {
	vector<int> contours_areas;

	sort(contours.begin(), contours.end(), [](vector<Point> cntr_a, vector<Point> cntr_b) {
		return contourArea(cntr_a) > contourArea(cntr_b);
		});
}

vector<vector<Point>> approxContours(vector<vector<Point>> contours) {
	vector<vector<Point>> poly_contours(contours.size());
	int idx = 0;

	for_each(contours.begin(), contours.end(), [&](vector<Point> cntr) {
		double eps = arcLength(cntr, true) * 0.02;
		approxPolyDP(cntr, poly_contours[idx], eps, true);
		++idx;
	});

	return poly_contours;
}

vector<Point> getContours(Mat img) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	contours = approxContours(contours);
	getSortedAreas(contours);

	return contours[0];
}

vector<Point> reorder(vector<Point> points) {
	vector<Point> reordered_points;
	vector<int> sum_points, sub_points;

	for_each(points.begin(), points.end(), [&](Point pt) {
		sum_points.push_back(pt.x + pt.y);
		sub_points.push_back(pt.x - pt.y);
	});

	reordered_points.push_back(points[min_element(sub_points.begin(), sub_points.end()) - sub_points.begin()]);
	reordered_points.push_back(points[min_element(sum_points.begin(), sum_points.end()) - sum_points.begin()]);
	reordered_points.push_back(points[max_element(sum_points.begin(), sum_points.end()) - sum_points.begin()]);
	reordered_points.push_back(points[max_element(sub_points.begin(), sub_points.end()) - sub_points.begin()]);

	return reordered_points;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h) {
	Point2f src[4] = { points[0], points[1], points[2], points[3] };
	Point2f dst[4] = { {0.0f, 0.0f}, {0.0f, h}, {w, 0.0f}, {w, h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	
	Mat warp;
	warpPerspective(img, warp, matrix, Point(w, h));

	return warp;
}

pair<int, int> getSize(vector<Point> points) {
	return make_pair(points[2].x - points[1].x, points[2].y - points[1].y);
}

int main() {
	Mat img = imread("Resources/2.jpg");
	resize(img, img, Size(), 0.3, 0.3);

	Mat dil = prepareImage(img);

	vector<Point> main_contour = getContours(dil);
	main_contour = reorder(main_contour);

	pair<int, int> img_size = getSize(main_contour);
	Mat warp = getWarp(img, main_contour, img_size.first, img_size.second);

	rotate(warp, warp, ROTATE_180);
	flip(warp, warp, 1);

	imshow("Scan", warp);
	waitKey(0);
}