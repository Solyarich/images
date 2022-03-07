#include <opencv2/opencv.hpp>
#include <cmath>

int func(int x) {
  double param = 0;
  int ans = 0;
  param = 10 * sin(x / 8) + x;
  ans = floor(param);
  if (ans > 255) {
    return 255;
  }
  if (ans < 0) {
    return 0;
  }
  return ans;
}

int main() {
  std::string name = "../../../data/cross_0256x0256.png";

  cv::Mat img = cv::imread(name, CV_8UC1);

  cv::Mat graph(512, 512, CV_8UC1);
  graph = 255;
  for (int i = 0; i < 255; i++) {
    graph.at<int8_t>(i, func(i)) = 0;
  }
  cv::imwrite("lab01.png", img);
  cv::imwrite("lab02.png", graph);

}