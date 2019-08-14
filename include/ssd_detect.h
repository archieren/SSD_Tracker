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
#include "tensorflow/cc/ops/image_ops.h"
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
    Detector(const std::string& model_file,const std::string& label_file);
    ~Detector(){};
    std::vector<std::vector<float> > detect(const cv::Mat& img);

  private:
    std::unique_ptr<tensorflow::Session> sess;
    std::map<int,std::string> labelsMap ;
    const std::string inputLayer = "image_tensor:0";
    const std::vector<std::string>  outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};
};

#endif //PROJECT_SSD_DETECT_H
