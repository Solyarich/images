#include <opencv2/opencv.hpp>

int main()
{
  cv::Mat img(450, 300, CV_32FC1);

  for (int i = 0; i < 150; i++)
  {
    for (int j = 0; j < 150; j++)
    {
      img.at<float>(i, j) = 0;
      img.at<float>(i, j + 150) = 255 / 255.0;
      img.at<float>(i + 150, j) = 127 / 255.0;
      img.at<float>(i + 150, j + 150) = 0;
      img.at<float>(i + 300, j) = 255 / 255.0;
      img.at<float>(i + 300, j + 150) = 127 / 255.0;
    }

  }
  cv::circle(img, cv::Point(75, 75), 50, 255 / 255.0, cv::FILLED);
  cv::circle(img, cv::Point(75, 75 + 150), 50, 0, cv::FILLED);
  cv::circle(img, cv::Point(75, 75 + 300), 50, 0, cv::FILLED);
  cv::circle(img, cv::Point(75 + 150, 75), 50, 127 / 255.0, cv::FILLED);
  cv::circle(img, cv::Point(75 + 150, 75 + 150), 50, 127 / 255.0, cv::FILLED);
  cv::circle(img, cv::Point(75 + 150, 75 + 300), 50, 255 / 255.0, cv::FILLED);

  cv::Mat I_1(2, 2, CV_32S);
  I_1.at<int>(0, 0) = 1;
  I_1.at<int>(0, 1) = 0;
  I_1.at<int>(1, 0) = 0;
  I_1.at<int>(1, 1) = -1;

  cv::Mat task_I_1;

  cv::filter2D(img, task_I_1, -1, I_1, cv::Point(0, 0));

  cv::Mat I_2(2, 2, CV_32S);
  I_2.at<int>(0, 0) = 0;
  I_2.at<int>(0, 1) = 1;
  I_2.at<int>(1, 0) = -1;
  I_2.at<int>(1, 1) = 0;

  cv::Mat task_I_2;
  cv::filter2D(img, task_I_2, -1, I_2, cv::Point(0, 0));

  cv::Mat img_middle(img.size(), CV_32FC1);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      img_middle.at<float>(i, j) = std::sqrt(task_I_1.at<float>(i, j) * task_I_1.at<float>(i, j) + task_I_2.at<float>(i, j) * task_I_2.at<float>(i, j));
    }
  }

  task_I_1 = (task_I_1 + 1) / 2;
  task_I_2 = (task_I_2 + 1) / 2;

  img = img * 255;
  task_I_1 = task_I_1 * 255;
  task_I_2 = task_I_2 * 255;
  img_middle = img_middle * 255;

  cv::imwrite("task1.png", img);
  cv::imwrite("task2.png", task_I_1);
  cv::imwrite("task3.png", task_I_2);
  cv::imwrite("task4.png", img_middle);

  return 0;
}