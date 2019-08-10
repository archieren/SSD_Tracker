#ifndef CTRACKER_H
#define CTRACKER_H
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <deque>

#include "defines.h"
#include "tracker/track.h"
#include "tracker/LocalTracker.h"

struct TrackerSettings
{
    /// Use local tracking for regions between two frames
    /// It was coined for tracking small and slow objects: key points on objects tracking with LK optical flow
    /// The most applications don't need this parameter
    ///
    bool m_useLocalTracking = false;

    tracking::DistType m_distType = tracking::DistCenters;
    tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
    tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
    tracking::LostTrackType m_lostTrackType = tracking::TrackKCF;
    tracking::MatchType m_matchType = tracking::MatchHungrian;

    /// Time step for Kalman
    track_t m_dt = 1.0f;

    /// Noise magnitude for Kalman
    track_t m_accelNoiseMag = 0.1f;

    /// Distance threshold for Assignment problem
    // for tracking::DistCenters or for tracking::DistRects
    // (for tracking::DiscJaccard it need from 0 to 1)
    track_t m_distThres = 50;

    /// If the object don't assignment more than this frames then it will be removed
    size_t m_maximumAllowedSkippedFrames = 25;

    /// The maximum trajectory length
    size_t m_maxTraceLength = 50;

    /// Detection abandoned objects
    bool m_useAbandonedDetection = false;

    /// After this time (in seconds) the object is considered abandoned
    int m_minStaticTime = 5;

    /// After this time (in seconds) the abandoned object will be removed
    int m_maxStaticTime = 25;
};

class CTracker
{
public:
    CTracker(const TrackerSettings& settings);
	~CTracker(void);

    tracks_t tracks;
    void Update(const regions_t& regions, cv::UMat grayFrame, float fps);

    bool GrayFrameToTrack() const
    {
        return m_settings.m_lostTrackType != tracking::LostTrackType::TrackGOTURN;
    }

private:
    TrackerSettings m_settings;

    size_t m_nextTrackID;

    LocalTracker m_localTracker;

    cv::UMat m_prevFrame;

    void UpdateHungrian(const regions_t& regions, cv::UMat grayFrame, float fps);
};
#endif
