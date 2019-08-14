#include "ssd_detect.h"
#include "utils.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;
using tensorflow::uint8;

Detector::Detector(
    const std::string& model_file,
    const std::string& label_file) {
        LoadPbModel_(model_file,label_file,0.2f,&sess,&labelsMap);
}

std::vector<std::vector<float> >  Detector::detect(const cv::Mat& img) {
    std::vector<std::vector<float> > detections;
    std::vector<Tensor> results;
    Tensor img_tensor = readTensorFromMat(img);
    Status status = sess->Run({{inputLayer,img_tensor}},outputLayer,{},&results);
    if (! status.ok()){
        return detections;
    }
    tensorflow::TTypes<float>::Flat scores = results[1].flat<float>();
	tensorflow::TTypes<float>::Flat classes = results[2].flat<float>();
	tensorflow::TTypes<float>::Flat numDetections = results[3].flat<float>();
	tensorflow::TTypes<float, 3>::Tensor boxes = results[0].flat_outer_dims<float, 3>();
    for(size_t k=0;k < scores.size();k++){
        //Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        std::vector<float> detection;
    }


    return detections;
}
