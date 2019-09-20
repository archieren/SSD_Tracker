#include <opencv2/opencv.hpp>
#include "defines.h"
#include "ssd_detect.h"
#include "tracker/Ctracker.h"

class PipeLine{
public:
    PipeLine(const std::string& model_file,const std::string& label_file){
        m_detector = std::make_unique<Detector>(model_file,label_file);

        TrackerSettings settings;
        settings.m_useLocalTracking = false;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchBipart;        // tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 100;                          // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = (size_t)(1 * m_fps);  // Maximum allowed skipped frames
        settings.m_maxTraceLength = (size_t)(10 * m_fps);

        m_tracker = std::make_unique<CTracker>(settings);

        // Different color used for path lines in tracking
        m_colors.emplace_back(cv::Scalar(255, 0, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 255));
        m_colors.emplace_back(cv::Scalar(255, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 127, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 127));
    }

    void process(std::string video_file,std::string out_file){
        cv::VideoCapture cap(video_file);
        if (! cap.isOpened()){
            LOG(FATAL) << "Failed to open video :" << video_file;
        }
        else {
            LOG(INFO) << "Video :" << video_file << " opened!";
        }
        auto frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        auto frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::Mat frame;
        int frameCount = 0;

        // video output
        cv::VideoWriter writer;
        writer.open(out_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), m_fps, cv::Size(frame_width, frame_height), true);


        while (true) {
            if (! cap.read(frame)) break;
            regions_t detections=m_detector->detect(frame);
            cv::UMat clFrame;
            clFrame = frame.getUMat(cv::ACCESS_READ);
            m_tracker->Update(detections, clFrame, m_fps);
            DrawData(frame);
            writer << frame;
            ++frameCount;
        }
    }

protected:
    void DrawTrack(cv::Mat frame,int resizeCoeff,const CTrack& track,bool drawTrajectory = true){
        auto ResizeRect = [&](const cv::Rect& r) -> cv::Rect
        {
            return cv::Rect(resizeCoeff * r.x, resizeCoeff * r.y, resizeCoeff * r.width, resizeCoeff * r.height);
        };
        auto ResizePoint = [&](const cv::Point& pt) -> cv::Point
        {
            return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
        };
        cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

        cv::rectangle(frame, ResizeRect(track.GetLastRect()), cl, 2, cv::LINE_AA);

        if (drawTrajectory)
        {
            for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
            {
                const TrajectoryPoint& pt1 = track.m_trace.at(j);
                const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
                cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 4, cv::LINE_AA);
            }
        }
    }

    void DrawData(cv::Mat frame){
        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
                //+ track->m_lastRegion.m_type
                std::string label = std::to_string(track->m_trackID)  + ": " + std::to_string((int)(track->m_lastRegion.m_confidence * 100)) + " %";
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame,
                              cv::Rect(cv::Point(rect.x, rect.y - labelSize.height - baseLine), cv::Size(labelSize.width, labelSize.height + baseLine)),
                              cv::Scalar(255, 255, 255));
                cv::putText(frame, label, cv::Point(rect.x, rect.y - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0),2);
            }
        }

    }

private:
    std::vector<cv::Scalar> m_colors;
    std::unique_ptr<Detector> m_detector;
    std::unique_ptr<CTracker> m_tracker;
    float m_fps = 30;
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

    //std::string video_file="../data/video/(0).mp4";
    //std::string video_file="../data/video/pedestrian.mp4";
    //std::string video_file="../data/video/pedestrian.mp4";
    std::string video_file="../data/video/TownCentreXVID.avi";
    std::string outFile = "../data/video/out.mp4";
    pipeLine.process(video_file,outFile);
    // TODO: put these variables in main


    return 0;
}
