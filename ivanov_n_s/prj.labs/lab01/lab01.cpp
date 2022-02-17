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

  float start = clock();

  cv::Mat G_1(60, 768, CV_8UC1);
  I_1.convertTo(G_1, CV_32F, 1 / 255.0);
  cv::pow(G_1, kor, G_1);
  G_1.convertTo(G_1, CV_8UC1, 255);

  float time = clock() - start;

  float start_2 = clock();

  cv::Mat G_2(60, 768, CV_8UC1);
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 768; j++) {
      G_2.at<int8_t>(i, j) = pow((j / 3) / 255.0, kor) * 255.0;
    }
  }

  float time_2 = clock() - start_2;

  cv::Mat img(180, 768, CV_8UC1);
  I_1.copyTo(img(cv::Rect2d(0, 0, 768, 60)));
  G_1.copyTo(img(cv::Rect2d(0, 60, 768, 60)));
  G_2.copyTo(img(cv::Rect2d(0, 120, 768, 60)));

  std::cout << "Time pow: " << time << '\n';
  std::cout << "Time at: " << time_2 << '\n';

  cv::imwrite("lab01.png", img);
}
