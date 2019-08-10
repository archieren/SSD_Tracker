#ifndef PROJECT_DEFINE_H
#define PROJECT_DEFINE_H
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define Mat_t CV_32FC

typedef std::map<std::string, std::string> config_t;

class CRegion
{
public:
    CRegion(): m_type(""), m_confidence(-1)
    {
    }

    CRegion(const cv::Rect& rect)
            : m_rect(rect)
    {

    }

    CRegion(const cv::Rect& rect, const std::string& type, float confidence)
            : m_rect(rect), m_type(type), m_confidence(confidence)
    {

    }

    cv::Rect m_rect;
    std::vector<cv::Point2f> m_points;

    std::string m_type;
    float m_confidence;
};

typedef std::vector<CRegion> regions_t;

namespace tracking
{
    enum Detectors
    {
        Motion_VIBE,
        Motion_MOG,
        Motion_GMG,
        Motion_CNT,
        Motion_SuBSENSE,
        Motion_LOBSTER,
        Motion_MOG2,
        Face_HAAR,
        Pedestrian_HOG,
        Pedestrian_C4,
        SSD_MobileNet,
        Yolo
    };

    enum DistType
    {
        DistCenters = 0,
        DistRects = 1,
        DistJaccard = 2
    };

    enum FilterGoal
    {
        FilterCenter = 0,
        FilterRect = 1
    };

    enum KalmanType
    {
        KalmanLinear = 0,
        KalmanUnscented = 1,
        KalmanAugmentedUnscented
    };

    enum MatchType
    {
        MatchHungrian = 0,
        MatchBipart = 1
    };

    enum LostTrackType
    {
        TrackNone = 0,
        TrackKCF = 1,
        TrackMIL,
        TrackMedianFlow,
        TrackGOTURN,
        TrackMOSSE
    };
}

#endif //PROJECT_DEFINE_H
