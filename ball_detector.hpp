#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <vector>

struct BallDetectorConfig {
    // Expected ball radius in pixels — adjust to your camera/distance setup
    double minRadius = 15.0;
    double maxRadius = 200.0;

    // Ellipse area filter (π * a * b). Rejects ellipses outside this pixel range.
    double minEllipseArea = 900.0;
    double maxEllipseArea = 10000.0;

    // Ellipse axis ratio (minor/major): reject if major > 1.3 * minor (i.e. ratio < 1/1.3)
    double minAxisRatio = 1.0 / 1.3;
    double maxAxisRatio = 1.0;

    // Edge detection
    int    blurKernelSize  = 5;
    double cannyLow        = 20.0;
    double cannyHigh       = 60.0;

    // Ellipse fit quality: mean normalized algebraic residual per point.
    // Points on the ellipse yield 0; further away yields higher values.
    // Good range: 0.04 – 0.10 depending on image quality.
    double maxFitResidual  = 0.07;

    // Minimum contour points for a stable ellipse fit (OpenCV requires >= 5)
    int minContourPoints   = 12;

    // Interior uniformity: coefficient of variation (stddev/mean) of pixel
    // brightness inside the fitted ellipse. Balls are smoother than backgrounds
    // with windows or texture. Set to 1.0 to disable.
    double maxInteriorCV   = 0.25;

    // NMS: two ellipses are considered duplicates if their centers are closer
    // than this fraction of their average major axis. Set to 0 to disable.
    double nmsOverlapFraction = 0.5;
};

struct RansacConfig {
    // Number of iterations
    int    iterations      = 200;
    // Algebraic residual threshold below which a point counts as inlier
    double inlierThreshold = 0.05;
    // Minimum inliers required to accept a RANSAC model
    int    minInliers      = 8;
    // Maximum fraction of contour points that may be cut from each end [0, 0.5)
    double maxCutFraction  = 0.40;
};

// Optional focused search: re-runs contour detection inside a 1.5× enlarged
// crop around the previous frame's ellipse, RANSAC-trims each found contour,
// and merges the results with the global detection before NMS.
struct FocusedSearchConfig {
    bool         enabled       = false;
    double       cropScale     = 1.5;  // half-axes multiplied by this for crop size
    // Outer RANSAC: how many (cutStart, cutEnd) pairs to try per contour
    int          trimIterations = 50;
    RansacConfig ransac;                // inner ransacRefineEllipse config
};

struct TemplateConfig {
    bool   enabled        = false;
    // Template-match search region radius as a multiple of the template's half-size
    double searchScale    = 2.5;
    // Minimum TM_CCOEFF_NORMED score to accept a template match
    double matchThreshold = 0.5;
    // After a successful match, restrict ALL detection to this multiple of the
    // ellipse's half-axes around the match center
    double roiScale       = 3.0;
};

// Returns all contours that are consistent with a ball of the configured shape.
// Partial contours (arcs) are included if they fit an ellipse well.
// If focusCfg.enabled and prevEllipse is non-null, additionally searches a
// zoomed crop around the previous ellipse and merges those contours in.
// NMS is applied before returning to suppress duplicates.
std::vector<std::vector<cv::Point>> detectBallContours(
    const cv::Mat&             frame,
    const BallDetectorConfig&  cfg,
    const FocusedSearchConfig& focusCfg    = {},
    const cv::RotatedRect*     prevEllipse = nullptr);

// RANSAC-based ellipse refinement: removes outlier points from a contour,
// fits an ellipse to the inlier consensus set, and returns it.
// inliers is filled with the subset of contour points that support the model.
cv::RotatedRect ransacRefineEllipse(
    const std::vector<cv::Point>& contour,
    const RansacConfig& cfg,
    std::vector<cv::Point>& inliers);
