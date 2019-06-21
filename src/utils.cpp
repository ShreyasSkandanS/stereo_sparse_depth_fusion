#include "stereo_sdf/utils.h"

void Utils::convertCVMatToPNG(const cv::Mat &cvmat_img,
                              png::image<png::rgb_pixel> &png_img)
{
    png_img.resize(cvmat_img.cols, cvmat_img.rows);
    for (int r = 0; r < cvmat_img.rows; r++) {
        for (int c = 0; c < cvmat_img.cols; c++) {
            const cv::Vec3b& bgr_c = cvmat_img.at<cv::Vec3b>(r, c);
            png_img.set_pixel(c,
                              r,
                              png::rgb_pixel(bgr_c[2], bgr_c[1], bgr_c[0]));
        }
    }
}

cv::Mat Utils::convertFloatToCVMat(int width,
                                   int height,
                                   const float* data)
{
    cv::Mat1w result(height, width);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            result(y, x) = (ushort)(data[width*y + x] * 256.0f + 0.5);
        }
    }
    return result;
}

void Utils::visualizeDepthImage(cv::Mat &depth_image)
{
  cv::Mat img_disp;
  double min, max;
  cv::minMaxIdx(depth_image, &min, &max);
  img_disp = depth_image / (double)max;
  img_disp.convertTo(img_disp, CV_8UC1, 255, 0);
  cv::applyColorMap(img_disp, img_disp, cv::COLORMAP_JET);
  cv::imshow("Visualization", img_disp);
  cv::waitKey(3);
}


cv::Mat Utils::colorMapDisparity(cv::Mat &disp_image)
{
	disp_image = disp_image / (double)256.0;
  cv::Mat img_disp;
  disp_image.convertTo(img_disp, CV_8UC1);
  cv::applyColorMap(img_disp, img_disp, cv::COLORMAP_JET);
	return img_disp.clone();
}

void Utils::CVMatSave(const std::string &filename,
                      const cv::Mat &mat)
{
  std::ofstream fs(filename, std::fstream::binary);
  int type = mat.type();
  int channels = mat.channels();
  fs.write((char*)&mat.rows, sizeof(int));
  fs.write((char*)&mat.cols, sizeof(int));
  fs.write((char*)&type, sizeof(int));
  fs.write((char*)&channels, sizeof(int));
  if (mat.isContinuous()) {
    fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
  } else {
    int rowsz = CV_ELEM_SIZE(type) * mat.cols;
    for (int r = 0; r < mat.rows; ++r) {
      fs.write(mat.ptr<char>(r), rowsz);
    }
  }
}

cv::Mat Utils::CVMatLoad(const std::string &filename)
{
  std::ifstream fs(filename, std::fstream::binary);
  int rows, cols, type, channels;
  fs.read((char*)&rows, sizeof(int));
  fs.read((char*)&cols, sizeof(int));
  fs.read((char*)&type, sizeof(int));
  fs.read((char*)&channels, sizeof(int));
  cv::Mat mat(rows, cols, type);
  fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);
  return mat;
}

void Utils::calculateAccuracy(cv::Mat input,
                              cv::Mat ground_truth,
                              double &average_error,
                              cv::Mat_<double> &error_image,
                              cv::Mat sampling_mask)
{
  std::ignore = sampling_mask;
  cv::Mat_<double> errors = cv::Mat::zeros(input.rows, input.cols, CV_64F);
  double total_error = 0;
  int total_points = 0;
  for (int r = 0; r < input.rows; r++) {
    for (int c =0; c < input.cols; c++) {
      double lidar_pt = ground_truth.at<double>(r,c);
      double input_pt = input.at<double>(r,c);
      if (lidar_pt > 0) {
        double err = fabs(lidar_pt - input_pt);
        errors.at<double>(r,c) = err;
        total_error = total_error + err;
        total_points = total_points + 1;
      }
    }
  }
  average_error = total_error / (double)total_points;
  cv::Mat gt_mask;
  cv::threshold(ground_truth, gt_mask, 0, 1, cv::THRESH_BINARY);
  double gt_mask_sum = cv::sum(gt_mask)[0];
  std::cout << "Total ground truth points: " << gt_mask_sum << std::endl;
  error_image = errors.clone();
}

std::unordered_set<int> Utils::BobFloydAlgo(int sampleSize,
                                            int rangeUpperBound)
{
     std::unordered_set<int> sample;
     std::default_random_engine generator;
     for(int d = rangeUpperBound - sampleSize; d < rangeUpperBound; d++)
     {
           int t = std::uniform_int_distribution<>(0, d)(generator);
           if (sample.find(t) == sample.end() )
               sample.insert(t);
           else
               sample.insert(d);
     }
     return sample;
}

void Utils::generateRandomSamplingMask(cv::Mat depth_image,
                                       cv::Mat &sampling_mask,
                                       double sampling_fraction)
{
  cv::Mat depth_image_local = depth_image.clone();
  depth_image_local.convertTo(depth_image_local, CV_32F);
  cv::Mat lidar_mask;
  cv::threshold(depth_image_local, lidar_mask, 0, 255, cv::THRESH_BINARY);
  std::vector<cv::Point> gt_points;
  lidar_mask.convertTo(lidar_mask, CV_8UC1);
  cv::findNonZero(lidar_mask, gt_points);
  int total_pts = gt_points.size();
  cv::Mat sampled_mask = cv::Mat::zeros(depth_image.rows,
                                        depth_image.cols,
                                        CV_8UC1);
  int total_pts_to_sample = (int) (total_pts * sampling_fraction);
  std::unordered_set<int> sample_indexes = BobFloydAlgo(total_pts_to_sample,
                                                        total_pts);
	for (const auto& elem:sample_indexes) {
		sampled_mask.at<unsigned char>(gt_points[elem]) = 255;
	}
	sampling_mask = sampled_mask.clone();
}
