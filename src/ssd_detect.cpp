#include "ssd_detect.h"


void Detector::initDetection(const std::string& model_file,
                   const std::string& weights_file,
                   const std::string& mean_file,
                   const std::string& mean_value) {
}


std::vector<std::vector<float> > Detector::Detect(const cv::Mat& img) {
    std::vector<std::vector<float> > detections;
    return detections;
}



void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {

}


void Detector::Preprocess(const cv::Mat& img,
                          std::vector<cv::Mat>* input_channels) {

}

