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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"


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
  static Detector * ssd_Detector;
  static std::unique_ptr<tensorflow::Session> sess;
  static std::map<int,std::string> labelsMap ;
};

#endif //PROJECT_SSD_DETECT_H
