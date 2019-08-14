#include "utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>
#include <numeric>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>




using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
*/
Status loadGraph(const string &graph_file_name,
                 unique_ptr<tensorflow::Session> *sess)
{
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	sess->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = (*sess)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}

/** Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
*/
Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

	// Read file into a string
	ifstream t(fileName);
	if (t.bad())
		return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
	stringstream buffer;
	buffer << t.rdbuf();
	string fileString = buffer.str();

	// Search entry patterns of type 'item { ... }' and parse each of them
	smatch matcherEntry;
	smatch matcherId;
	smatch matcherName;
	const regex reEntry("item \\{([\\S\\s]*?)\\}");
	const regex reId("[0-9]+");
	const regex reName("\'.+\'");
	string entry;

	auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
	auto stringEnd = sregex_iterator();

	int id;
	string name;
	for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
		matcherEntry = *i;
		entry = matcherEntry.str();
		regex_search(entry, matcherId, reId);
		if (!matcherId.empty())
			id = stoi(matcherId[0].str());
		else
			continue;
		regex_search(entry, matcherName, reName);
		if (!matcherName.empty())
			name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
		else
			continue;
		labelsMap.insert(pair<int, string>(id, name));
	}
	return Status::OK();
}

int LoadPbModel_(std::string GraphPath,
                 std::string LabelsPath,
                 float MemoryUsage,
                 std::unique_ptr<tensorflow::Session> *CudaSession,
                 std::map<int, std::string> *LabelsMap)
{
	tensorflow::SessionOptions session_options;
	session_options.config.mutable_gpu_options()->set_allow_growth(false);
	session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(MemoryUsage);
	(*CudaSession).reset(tensorflow::NewSession(session_options));

	// Load and initialize the model from .pb file
	Status loadGraphStatus = loadGraph(GraphPath, CudaSession);
	if (!loadGraphStatus.ok()) {
		LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
		return 0;
	}
	else
		LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

	// Load labels map from .pbtxt file

	Status readLabelsMapStatus = readLabelsMapFile(LabelsPath, *LabelsMap);
	if (!readLabelsMapStatus.ok()) {
		LOG(ERROR) << "readLabelsMapFile(): ERROR";
		return 0;
	}
	else
		LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << (*LabelsMap).size() << " label(s)" << endl;
	return 1;
}

int LoadPbModel_WithNoLabel(std::string GraphPath,
                            float MemoryUsage,
                            std::unique_ptr<tensorflow::Session> *CudaSession) {
	tensorflow::SessionOptions session_options;
	session_options.config.mutable_gpu_options()->set_allow_growth(false);
	session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(MemoryUsage);
	(*CudaSession).reset(tensorflow::NewSession(session_options));

	// Load and initialize the model from .pb file
	Status loadGraphStatus = loadGraph(GraphPath, CudaSession);
	if (!loadGraphStatus.ok()) {
		LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
		return 0;
	}
	else {
		LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;
		return  1;
	}
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
*/
//Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {
//
//    auto root = tensorflow::Scope::NewRootScope();
//    using namespace ::tensorflow::ops;
//
//    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
//    float *p = outTensor.flat<float>().data();
//    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
//    mat.convertTo(fakeMat, CV_32FC3);
//
//    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
//    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
//    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);
//
//    // This runs the GraphDef network definition that we've just constructed, and
//    // returns the results in the output outTensor.
//    tensorflow::GraphDef graph;
//    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
//
//    vector<Tensor> outTensors;
//    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
//
//    TF_RETURN_IF_ERROR(session->Create(graph));
//    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));
//
//    outTensor = outTensors.at(0);
//    return Status::OK();
//}

/** Draw bounding box and add caption to the image.
*  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
*/
void drawBoundingBoxOnImage(Mat &image,
                            double yMin, double xMin, double yMax, double xMax,
                            double score, string label, bool scaled = true)
{
	cv::Point tl, br;
	if (scaled) {
		tl = cv::Point((int)(xMin * image.cols), (int)(yMin * image.rows));
		br = cv::Point((int)(xMax * image.cols), (int)(yMax * image.rows));
	}
	else {
		tl = cv::Point((int)xMin, (int)yMin);
		br = cv::Point((int)xMax, (int)yMax);
	}
	cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

	// Ceiling the score down to 3 decimals (weird!)
	float scoreRounded = floorf(score * 1000) / 1000;
	string scoreString = to_string(scoreRounded).substr(0, 5);
	string caption = label + " (" + scoreString + ")";

	// Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
	int fontCoeff = 12;
	cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
	cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
	cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
	cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
*  Box is drawn only if corresponding score is higher than the _threshold_.
*/
void drawBoundingBoxesOnImage(Mat &image,
	tensorflow::TTypes<float>::Flat &scores,
	tensorflow::TTypes<float>::Flat &classes,
	tensorflow::TTypes<float, 3>::Tensor &boxes,
	map<int, string> &labelsMap,
	vector<size_t> &idxs) {
	for (size_t j = 0; j < idxs.size(); j++)
		drawBoundingBoxOnImage(image,
			boxes(0, idxs.at(j), 0), boxes(0, idxs.at(j), 1),
			boxes(0, idxs.at(j), 2), boxes(0, idxs.at(j), 3),
			scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}

/** Calculate intersection-over-union (IOU) for two given bbox Rects.
*/
double IOU(Rect2f box1, Rect2f box2) {

	float xA = max(box1.tl().x, box2.tl().x);
	float yA = max(box1.tl().y, box2.tl().y);
	float xB = min(box1.br().x, box2.br().x);
	float yB = min(box1.br().y, box2.br().y);

	float w = max(0.0f, xB - xA);  // 原来这里是有问题的！
	float h = max(0.0f, yB - yA);

	float intersectArea = w*h;
	float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

	return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
*  and IOU <= thresholdIOU with others).
*/
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore)
{

	vector<size_t> sortIdxs(scores.size());
	iota(sortIdxs.begin(), sortIdxs.end(), 0);

	// Create set of "bad" idxs
	set<size_t> badIdxs = set<size_t>();
	size_t i = 0;
	while (i < sortIdxs.size()) {
		if (scores(sortIdxs.at(i)) < thresholdScore)
			badIdxs.insert(sortIdxs[i]);
		if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
			i++;
			continue;
		}

		Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
			Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
		for (size_t j = i + 1; j < sortIdxs.size(); j++) {
			if (scores(sortIdxs.at(j)) < thresholdScore) {
				badIdxs.insert(sortIdxs[j]);
				continue;
			}
			Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
				Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
			if (IOU(box1, box2) > thresholdIOU)
				badIdxs.insert(sortIdxs[j]);
		}
		i++;
	}

	// Prepare "good" idxs for return
	vector<size_t> goodIdxs = vector<size_t>();
	for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
		if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
			goodIdxs.push_back(*it);

	return goodIdxs;
}

/**  和上面的那个差不多，添加了按类来过滤的情况.
*/
std::vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &classes,
                                tensorflow::TTypes<float>::Flat &scores,
                                tensorflow::TTypes<float, 3>::Tensor &boxes,
                                string classFilter,
                                std::map<int, std::string> &labelsMap
                                , float ThresholdScore, float ThresholdIOU)
{

	vector<size_t> sortIdxs(scores.size());
	iota(sortIdxs.begin(), sortIdxs.end(), 0);
	// Create set of "bad" idxs
	set<size_t> badIdxs = set<size_t>();//“坏索引！”是个集合
	size_t i = 0;
	while (i < sortIdxs.size()) {
		if ((scores(sortIdxs.at(i)) < ThresholdScore) || (labelsMap[classes(sortIdxs.at(i))].find(classFilter) == string::npos)){//添加了一个过滤，按检测对象的分类符来过滤
			badIdxs.insert(sortIdxs.at(i));
			i++;
			continue;
		}
		if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {  // i处的内容，已标记为坏索引. 这是可能的，看下面和IOU有关的部分。
			i++;
			continue;
		}

		Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
							 Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2))
							);
		for (size_t j = i + 1; j < sortIdxs.size(); j++) {
			if ((scores(sortIdxs.at(j)) < ThresholdScore) || (labelsMap[classes(sortIdxs.at(i))].find(classFilter) == string::npos)) {
				badIdxs.insert(sortIdxs.at(j));
				continue;
			}
			Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
								 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2))
								);
			if (IOU(box1, box2) > ThresholdIOU)  // 原来这个IOU的算法是错误的。我修改了的！
				if (scores(sortIdxs.at(i)) >= scores(sortIdxs.at(j))){  //我修改了一点，重叠的同类对象，保留得分大的那个.
					badIdxs.insert(sortIdxs.at(j));
				}
				else{
					badIdxs.insert(sortIdxs.at(i));
				}

		}
		i++;
	}

	// Prepare "good" idxs for return
	vector<size_t> goodIdxs = vector<size_t>();
	for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
		if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end()) //如果sortIdxs.at(*it)不在坏索引中，则为好！
		{
			goodIdxs.push_back(*it);
		}

	return goodIdxs;
}

Tensor readTensorFromMat(const Mat &mat) {
	int height = mat.rows;
	int width = mat.cols;
	int depth = mat.channels();
	Tensor inputTensor(tensorflow::DT_UINT8, tensorflow::TensorShape({ 1, height, width, depth }));
	auto inputTensorMapped = inputTensor.tensor<tensorflow::uint8, 4>();

	cv::Mat frame;
	mat.convertTo(frame, CV_8UC3);
	const tensorflow::uint8* source_data = (tensorflow::uint8*)frame.data;
	for (int y = 0; y < height; y++) {
		const tensorflow::uint8* source_row = source_data + (y*width*depth);
		for (int x = 0; x < width; x++) {
			const tensorflow::uint8* source_pixel = source_row + (x*depth);
			for (int c = 0; c < depth; c++) {
				const tensorflow::uint8* source_value = source_pixel + c;
				inputTensorMapped(0, y, x, c) = *source_value;
			}
		}
	}
	return inputTensor;
}

Tensor readFloatTensorFromMat(const Mat &mat) {
	int height = mat.rows;
	int width = mat.cols;
	int depth = mat.channels();
	Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, height, width, depth }));
	auto inputTensorMapped = inputTensor.tensor<float, 4>();

	cv::Mat frame;
	mat.convertTo(frame, CV_32FC1);
	const float* source_data = (float*)frame.data;
	for (int y = 0; y < height; y++) {
		const float* source_row = source_data + (y*width*depth);
		for (int x = 0; x < width; x++) {
			const float* source_pixel = source_row + (x*depth);
			for (int c = 0; c < depth; c++) {
				const float* source_value = source_pixel + c;
				inputTensorMapped(0, y, x, c) = (*source_value)/255;
			}
		}
	}
	return inputTensor;
}
