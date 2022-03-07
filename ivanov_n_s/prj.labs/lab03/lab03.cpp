#include <opencv2/opencv.hpp>
#include <cmath>

uint8_t Bright(uint8_t x) {
	return cv::saturate_cast<uint8_t>(pow(x, x / 255.0));
}

int main() {

  std::string name = "../../../data/cross_0256x0256.png";

  cv::Mat img = cv::imread(name);
  cv::imwrite("lab03_rgb.png", img);

  cv::Mat img_gray(img.rows, img.cols, img.type());
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
  cv::imwrite("lab03_gre.png", img_gray);

  cv::Mat graph(img.rows, img.cols, CV_8UC1);
  graph = 255;

  for(int i = 0; i < graph.cols; ++i) {
    graph.at<uint8_t>(255 - Bright(i), i) = 0;
  }

  cv::resize(graph, graph, cv::Size(512, 512), 0, 0, cv::InterpolationFlags::INTER_CUBIC);
  cv::imwrite("lab03_viz_func.png", graph);

  cv::Mat lut(1, 256, CV_8UC1);
	for (int i = 0; i < lut.cols; i++) {
		lut.at<uint8_t>(0, i) = Bright(i);
	}

  cv::Mat graph_gray(img_gray.rows, img_gray.cols, img_gray.type());
  cv::LUT(img_gray, lut, graph_gray);
  cv::imwrite("lab03_gre_res.png", graph_gray);

  cv::Mat colors[3];

	cv::split(img, colors);

	cv::Mat bgr[3];
	colors[0].copyTo(bgr[0]);
	colors[1].copyTo(bgr[1]);
	colors[2].copyTo(bgr[2]);
		
	cv::Mat graph_colors[3];

	cv::LUT(bgr[0], lut, graph_colors[0]);
	cv::LUT(bgr[1], lut, graph_colors[1]);
	cv::LUT(bgr[2], lut, graph_colors[2]);

  cv::Mat img_gbrg[4];
  cv::Mat zeros = cv::Mat::zeros(cv::Size(img.rows, img.cols), CV_8UC1);
  cv::merge(std::vector<cv::Mat>({ graph_gray, graph_gray, graph_gray }), img_gbrg[0]);
  cv::merge(std::vector<cv::Mat>({ graph_colors[0], zeros, zeros }), img_gbrg[1]);
  cv::merge(std::vector<cv::Mat>({ zeros, graph_colors[1], zeros }), img_gbrg[2]);
  cv::merge(std::vector<cv::Mat>({ zeros, zeros, graph_colors[2] }), img_gbrg[3]);

  cv::Mat img_channels(img.rows * 2, img.cols * 2, img.type());
  cv::Rect2d rc(0, 0, img.cols, img.rows);
  img_gbrg[0].copyTo(img_channels(rc));
  rc.x += rc.width;
  img_gbrg[1].copyTo(img_channels(rc));
  rc.y += rc.height;
  img_gbrg[2].copyTo(img_channels(rc));
  rc.x -= rc.width;
  img_gbrg[3].copyTo(img_channels(rc));
  cv::imwrite("lab03_rgb_res.png", img_channels);

	return 0;
}