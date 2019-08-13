#include "ssd_detect.h"


using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;
using tensorflow::uint8;

Status LoadGraph(const std::string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

Detector* Detector::getDetector(
    const std::string& model_file,
    const std::string& label_file) {
  if (! ssd_Detector){
    LoadGraph(model_file,&sess);
  }
}


std::vector<std::vector<float> > Detector::Detect(const cv::Mat& img) {
    std::vector<std::vector<float> > detections;
    return detections;
}
