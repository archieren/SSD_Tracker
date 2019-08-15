#include <opencv2/opencv.hpp>
#include "defines.h"
#include "ssd_detect.h"
#include "tracker/Ctracker.h"

class PipeLine{
public:
    PipeLine(const std::string& model_file,const std::string& label_file){
        m_detector = std::make_unique<Detector>(model_file,label_file);

        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 100;                          // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = (size_t)(1 * m_fps);  // Maximum allowed skipped frames
        settings.m_maxTraceLength = (size_t)(5 * m_fps);

        m_tracker = std::make_unique<CTracker>(settings);
    }
private:
    std::unique_ptr<Detector> m_detector;
    std::unique_ptr<CTracker> m_tracker;
    float m_fps=30;
};

// ----------------------------------------------------------------------

const char* keys =
        {
                "{help h usage ?  |                    | Print usage| }"
        };

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string model_file="../models/frozen_inference_graph.pb";
    std::string label_file="../models/label_map.pbtxt";
    PipeLine  pipeLine(model_file,label_file);

    std::string video_file="../data/video/TownCentreXVID.avi";

    // TODO: put these variables in main


    return 0;
}
