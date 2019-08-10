#ifndef TRACK_H
#define TRACK_H
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>
#include <opencv2/tracking.hpp>

#include "defines.h"
#include "tracker/Kalman.h"

/// The TrajectoryPoint struct
struct TrajectoryPoint
{

    TrajectoryPoint()
        : m_hasRaw(false)
    {
    }

    TrajectoryPoint(const Point_t& prediction)
        :
          m_hasRaw(false),
          m_prediction(prediction)
    {
    }

    TrajectoryPoint(const Point_t& prediction, const Point_t& raw)
        :
          m_hasRaw(true),
          m_prediction(prediction),
          m_raw(raw)
    {
    }

    bool m_hasRaw;
    Point_t m_prediction;
    Point_t m_raw;
};

class Trace
{
public:

    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }

    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }

    const TrajectoryPoint& at(size_t i) const
    {
        return m_trace[i];
    }

    size_t size() const
    {
        return m_trace.size();
    }

    void push_back(const Point_t& prediction)
    {
        m_trace.push_back(TrajectoryPoint(prediction));
    }
    void push_back(const Point_t& prediction, const Point_t& raw)
    {
        m_trace.push_back(TrajectoryPoint(prediction, raw));
    }

    void pop_front(size_t count)
    {
        if (count < size())
        {
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        }
        else
        {
            m_trace.clear();
        }
    }

    size_t GetRawCount(size_t lastPeriod) const
    {
        size_t res = 0;

        size_t i = 0;
        if (lastPeriod < m_trace.size())
        {
            i = m_trace.size() - lastPeriod;
        }
        for (; i < m_trace.size(); ++i)
        {
            if (m_trace[i].m_hasRaw)
            {
                ++res;
            }
        }

        return res;
    }

    void FirstPass()
    {
        m_firstPass = true;
    }

    void SecondPass()
    {
        m_secondPass = true;
    }

    bool GetFirstPass()
    {
        return m_firstPass;
    }

    bool GetSecondPass()
    {
        return m_secondPass;
    }


    bool m_firstPass = false;
    bool m_secondPass = false;
    bool m_directionFromLeft = false;
private:
    std::deque<TrajectoryPoint> m_trace;
};

class CTrack
{
public:
    CTrack(const CRegion& region,
            tracking::KalmanType kalmanType,
            track_t deltaTime,
            track_t accelNoiseMag,
            size_t trackID,
            bool filterObjectSize,
            tracking::LostTrackType externalTrackerForLost);

    track_t CalcDist(const Point_t& pt) const;
    track_t CalcDist(const cv::Rect& r) const;
    track_t CalcDistJaccard(const cv::Rect& r) const;

    bool CheckType(const std::string& type) const;

    void Update(const CRegion& region, bool dataCorrect, size_t max_trace_length, cv::UMat prevFrame, cv::UMat currFrame, int trajLen);

    bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const;
    bool IsStatic() const;
    bool IsStaticTimeout(int framesTime) const;

    Trace m_trace;
    size_t m_trackID;
    size_t m_skippedFrames;
    CRegion m_lastRegion;
    Point_t m_averagePoint;   ///< Average point after LocalTracking
    cv::Rect m_boundidgRect;  ///< Bounding rect after LocalTracking
    bool m_firstPass;
    bool m_secondPass;

    cv::Rect GetLastRect() const;

private:
    Point_t m_predictionPoint;
    cv::Rect m_predictionRect;
    TKalmanFilter* m_kalman;
    bool m_filterObjectSize;
    bool m_outOfTheFrame;

    tracking::LostTrackType m_externalTrackerForLost;
    cv::Ptr<cv::Tracker> m_tracker;
    void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);
    void CreateExternalTracker();
    void PointUpdate(const Point_t& pt, bool dataCorrect, const cv::Size& frameSize);
    bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region);
    bool m_isStatic = false;
    int m_staticFrames = 0;
    cv::UMat m_staticFrame;
    cv::Rect m_staticRect;
};
typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
#endif