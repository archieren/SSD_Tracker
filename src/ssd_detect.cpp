#include "ssd_detect.h"
#include "utils.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;
using tensorflow::uint8;

Detector::Detector(
    const std::string& model_file,
    const std::string& label_file) {
        LoadPbModel_(model_file,label_file,memmoryUsage,&sess,&labelsMap);
}

regions_t  Detector::detect(const cv::Mat& img) {
    int64 height = img.rows, width = img.cols;

    regions_t detections;
    detections.clear();
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

	std::vector<size_t> goodIdxs = filterBoxes(classes,scores, boxes,filter,labelsMap, thresholdScore,thresholdIOU );
	for (size_t i = 0; i < goodIdxs.size(); i++) { 
		auto ymin = static_cast<int>(boxes(0, goodIdxs.at(i), 0)*height);
		auto xmin = static_cast<int>(boxes(0, goodIdxs.at(i), 1)*width);
		auto ymax = static_cast<int>(boxes(0, goodIdxs.at(i), 2)*height);
		auto xmax = static_cast<int>(boxes(0, goodIdxs.at(i), 3)*width);
	    auto score = scores(goodIdxs.at(i));
		auto label = labelsMap[classes(goodIdxs.at(i))];
        cv::Rect object(xmin,ymin,xmax-xmin,ymax-ymin);
        detections.push_back(CRegion(object,label,score));
    }

    return detections;
}
