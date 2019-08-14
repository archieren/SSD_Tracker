#ifndef TF_DETECTOR_EXAMPLE_UTILS_H
#define TF_DETECTOR_EXAMPLE_UTILS_H

#include <vector>
#include <map>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/core/mat.hpp>


using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;


Status readLabelsMapFile(const string &fileName, std::map<int, string> &labelsMap);

Status loadGraph(const string &graph_file_name,
	std::unique_ptr<tensorflow::Session> *session);

int LoadPbModel_(std::string GraphPath,
                 std::string LabelsPath,
                 float MemoryUsage,
                 std::unique_ptr<tensorflow::Session> *CudaSession,
                 std::map<int, std::string> *LabelsMap);
int LoadPbModel_WithNoLabel(std::string GraphPath,
                            float MemoryUsage,
                            std::unique_ptr<tensorflow::Session> *CudaSession);
//Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor);

void drawBoundingBoxOnImage(cv::Mat &image,
                            double xMin, double yMin,double xMax, double yMax,
                            double score,
                            std::string label,
                            bool scaled);

void drawBoundingBoxesOnImage(cv::Mat &image,
	tensorflow::TTypes<float>::Flat &scores,
	tensorflow::TTypes<float>::Flat &classes,
	tensorflow::TTypes<float, 3>::Tensor &boxes,
	std::map<int, string> &labelsMap,
	std::vector<size_t> &idxs);

double IOU(cv::Rect box1, cv::Rect box2);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
	tensorflow::TTypes<float, 3>::Tensor &boxes,
	double thresholdIOU, double thresholdScore);

std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &classes,
	tensorflow::TTypes<float>::Flat &scores,
	tensorflow::TTypes<float, 3>::Tensor &boxes,
	string classFilter, std::map<int, std::string> &labelsMap
    , float ThresholdScore, float ThresholdIOU);

Tensor readTensorFromMat(const cv::Mat &mat);

Tensor readFloatTensorFromMat(const cv::Mat &mat);
#endif //TF_DETECTOR_EXAMPLE_UTILS_H
