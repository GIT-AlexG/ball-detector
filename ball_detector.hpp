#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <vector>

struct BallDetectorConfig {
    // Expected ball radius in pixels — adjust to your camera/distance setup
    double minRadius = 15.0;
    double maxRadius = 200.0;

    // Ellipse axis ratio (minor/major): 1.0 = circle; lower = oblique projection allowed
    double minAxisRatio = 0.45;
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

// Returns all contours that are consistent with a ball of the configured shape.
// Partial contours (arcs) are included if they fit an ellipse well.
std::vector<std::vector<cv::Point>> detectBallContours(
    const cv::Mat& frame,
    const BallDetectorConfig& cfg);

// RANSAC-based ellipse refinement: removes outlier points from a contour,
// fits an ellipse to the inlier consensus set, and returns it.
// inliers is filled with the subset of contour points that support the model.
cv::RotatedRect ransacRefineEllipse(
    const std::vector<cv::Point>& contour,
    const RansacConfig& cfg,
    std::vector<cv::Point>& inliers);
