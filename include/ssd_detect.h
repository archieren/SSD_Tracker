//
// Created by sumbal on 20/06/18.
//

#ifndef PROJECT_SSD_DETECT_H
#define PROJECT_SSD_DETECT_H

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// This is a demo code for using a SSD model to do detection.

class Detector {
 public:
    void initDetection(const std::string& model_file,
             const std::string& label_file);

    std::vector<std::vector<float> > Detect(const cv::Mat& img);


 private:
    // shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

#endif //PROJECT_SSD_DETECT_H
