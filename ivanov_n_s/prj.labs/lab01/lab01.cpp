#include <opencv2/opencv.hpp>

int main() {
  cv::Mat I_1(60, 768, CV_8UC1);
  I_1 = 0;
  double kor = 2.3;

  for (int i = 0; i < 60; i++) {
    int k = 0;
    for (int j = 0; j < 768; j += 3) {
      I_1.at<int8_t>(i, j) = k;
      I_1.at<int8_t>(i, j + 1) = k;
      I_1.at<int8_t>(i, j + 2) = k;
      k++;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  cv::Mat G_1(60, 768, CV_8UC1);
  I_1.convertTo(G_1, CV_32F, 1 / 255.0);
  cv::pow(G_1, kor, G_1);
  G_1.convertTo(G_1, CV_8UC1, 255);

  auto time = std::chrono::high_resolution_clock::now();
  auto res = std::chrono::duration_cast<std::chrono::microseconds>(time - start);
  auto start_2 = std::chrono::high_resolution_clock::now();

  cv::Mat G_2(60, 768, CV_8UC1);
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 768; j++) {
      G_2.at<int8_t>(i, j) = pow((j / 3) / 255.0, kor) * 255.0;
    }
  }

  auto time_2 = std::chrono::high_resolution_clock::now();
  auto res_2 = std::chrono::duration_cast<std::chrono::microseconds>(time_2 - start_2);

  cv::Mat img(180, 768, CV_8UC1);
  I_1.copyTo(img(cv::Rect2d(0, 0, 768, 60)));
  G_1.copyTo(img(cv::Rect2d(0, 60, 768, 60)));
  G_2.copyTo(img(cv::Rect2d(0, 120, 768, 60)));

  std::cout << "Time (pow): " << res.count() << " microseconds" << '\n';
  std::cout << "Time (at): " << res_2.count() << " microseconds" << '\n';

  cv::imwrite("lab01.png", img);
}
