#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include <png++/png.hpp>
#include <unordered_set>
#include <random>

class Utils
{
 public:
  void convertCVMatToPNG(const cv::Mat &cvmat_img,
                         png::image<png::rgb_pixel> &png_img);
  cv::Mat convertFloatToCVMat(int width,
                              int height,
                              const float* data);
  void visualizeDepthImage(cv::Mat &depth_image);
  cv::Mat colorMapDisparity(cv::Mat &disp_image);
  void CVMatSave(const std::string &filename,
                 const cv::Mat &mat);
  cv::Mat CVMatLoad(const std::string &filename);
  void calculateAccuracy(cv::Mat input,
                         cv::Mat gt,
                         double &error,
                         cv::Mat_<double> &err,
                         cv::Mat sampling_mask);
  std::unordered_set<int> BobFloydAlgo(int sampleSize,
                                       int rangeUpperBound);
  void generateRandomSamplingMask(cv::Mat depth_image,
                                  cv::Mat &sampling_mask,
                                  double sampling_fraction);
};
