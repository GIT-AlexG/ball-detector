#include "ball_detector.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

// Normalized algebraic residual of a point relative to a fitted ellipse.
// Returns |( x'/a )^2 + ( y'/b )^2 - 1| where x', y' are point coordinates
// in the ellipse-local frame (rotated and centred). Scale-independent.
static double ellipseResidual(const cv::Point2f& pt, const cv::RotatedRect& el) {
    float a = el.size.width  * 0.5f;
    float b = el.size.height * 0.5f;
    if (a < 1.f || b < 1.f) return 1.0;

    float angleRad = el.angle * static_cast<float>(CV_PI) / 180.f;
    float cosA = std::cos(-angleRad);
    float sinA = std::sin(-angleRad);

    float dx = pt.x - el.center.x;
    float dy = pt.y - el.center.y;

    float xr =  cosA * dx - sinA * dy;
    float yr =  sinA * dx + cosA * dy;

    float val = (xr / a) * (xr / a) + (yr / b) * (yr / b);
    return std::abs(val - 1.0f);
}

// Mean normalized algebraic residual over all contour points.
static double meanResidual(const std::vector<cv::Point>& contour,
                           const cv::RotatedRect& ellipse) {
    double sum = 0.0;
    for (const auto& p : contour)
        sum += ellipseResidual(cv::Point2f(static_cast<float>(p.x),
                                           static_cast<float>(p.y)), ellipse);
    return sum / contour.size();
}

// Coefficient of variation (stddev / mean) of brightness inside an ellipse mask.
static double interiorCV(const cv::Mat& gray, const cv::RotatedRect& el) {
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    cv::ellipse(mask, el, cv::Scalar(255), -1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev, mask);

    if (mean[0] < 1.0) return 1.0;
    return stddev[0] / mean[0];
}

std::vector<std::vector<cv::Point>> detectBallContours(
    const cv::Mat& frame,
    const BallDetectorConfig& cfg)
{
    // 1. Convert to grayscale and blur to suppress JPEG block artifacts
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    int k = cfg.blurKernelSize | 1;
    cv::GaussianBlur(gray, blurred, {k, k}, 0);

    // 2. Canny edge detection
    cv::Mat edges;
    cv::Canny(blurred, edges, cfg.cannyLow, cfg.cannyHigh);

    // 3. Find all contours (no hierarchy needed)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> result;
    result.reserve(8);

    for (auto& contour : contours) {
        if (static_cast<int>(contour.size()) < cfg.minContourPoints)
            continue;

        // 4. Fit ellipse (AMS is more robust to noise than the basic DLS fit)
        cv::RotatedRect el = cv::fitEllipseAMS(contour);

        float majorAxis = std::max(el.size.width, el.size.height) * 0.5f;
        float minorAxis = std::min(el.size.width, el.size.height) * 0.5f;

        // 5a. Size filter
        if (majorAxis < cfg.minRadius || majorAxis > cfg.maxRadius)
            continue;

        // 5b. Axis ratio filter: rejects lines and highly elongated shapes
        double axisRatio = (majorAxis > 0) ? (minorAxis / majorAxis) : 0.0;
        if (axisRatio < cfg.minAxisRatio || axisRatio > cfg.maxAxisRatio)
            continue;

        // 5c. Ellipse fit quality: contour points must lie close to the ellipse
        double residual = meanResidual(contour, el);
        if (residual > cfg.maxFitResidual)
            continue;

        // 5d. Interior uniformity: ball surface is smoother than structured backgrounds
        if (cfg.maxInteriorCV < 1.0) {
            double cv = interiorCV(gray, el);
            if (cv > cfg.maxInteriorCV)
                continue;
        }

        result.push_back(std::move(contour));
    }

    return result;
}

cv::RotatedRect ransacRefineEllipse(
    const std::vector<cv::Point>& contour,
    const RansacConfig& cfg,
    std::vector<cv::Point>& inliers)
{
    const int n = static_cast<int>(contour.size());
    constexpr int kSample = 5; // minimum points required by fitEllipseAMS

    if (n < kSample) {
        inliers = contour;
        return cv::fitEllipseAMS(contour);
    }

    std::mt19937 rng(42);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    int bestCount = 0;
    cv::RotatedRect bestEl;
    std::vector<cv::Point> sample(kSample);

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        // Draw kSample distinct random points (partial Fisher–Yates)
        for (int i = 0; i < kSample; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            std::swap(idx[i], idx[dist(rng)]);
            sample[i] = contour[idx[i]];
        }

        cv::RotatedRect el = cv::fitEllipseAMS(sample);

        // Count how many contour points lie within the inlier threshold
        int count = 0;
        for (const auto& p : contour)
            if (ellipseResidual(cv::Point2f(static_cast<float>(p.x),
                                            static_cast<float>(p.y)), el)
                    < cfg.inlierThreshold)
                ++count;

        if (count > bestCount) {
            bestCount = count;
            bestEl    = el;
        }
    }

    // Collect inliers for best model and refit
    inliers.clear();
    for (const auto& p : contour)
        if (ellipseResidual(cv::Point2f(static_cast<float>(p.x),
                                        static_cast<float>(p.y)), bestEl)
                < cfg.inlierThreshold)
            inliers.push_back(p);

    if (static_cast<int>(inliers.size()) >= cfg.minInliers)
        return cv::fitEllipseAMS(inliers);

    // Fallback: not enough inliers, return original fit
    inliers = contour;
    return cv::fitEllipseAMS(contour);
}
