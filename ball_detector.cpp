#include "ball_detector.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>

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

// Greedy NMS: sort by major axis (largest first), suppress any later ellipse
// whose center falls within overlapFraction * avg(major_a, major_b) of an
// already-accepted one.
static std::vector<std::vector<cv::Point>> applyNMS(
    const std::vector<std::vector<cv::Point>>& contours,
    double overlapFraction)
{
    if (contours.empty() || overlapFraction <= 0.0) return contours;

    struct Entry { int idx; cv::RotatedRect el; float major; };
    std::vector<Entry> entries;
    entries.reserve(contours.size());
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (contours[i].size() < 5) continue;
        cv::RotatedRect el = cv::fitEllipseAMS(contours[i]);
        float major = std::max(el.size.width, el.size.height) * 0.5f;
        entries.push_back({i, el, major});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.major > b.major; });

    std::vector<bool> suppressed(entries.size(), false);
    for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
        if (suppressed[i]) continue;
        for (int j = i + 1; j < static_cast<int>(entries.size()); ++j) {
            if (suppressed[j]) continue;
            float dx = entries[i].el.center.x - entries[j].el.center.x;
            float dy = entries[i].el.center.y - entries[j].el.center.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            float threshold = (entries[i].major + entries[j].major) * 0.5f
                              * static_cast<float>(overlapFraction);
            if (dist < threshold)
                suppressed[j] = true;
        }
    }

    std::vector<std::vector<cv::Point>> out;
    out.reserve(entries.size());
    for (int i = 0; i < static_cast<int>(entries.size()); ++i)
        if (!suppressed[i])
            out.push_back(contours[entries[i].idx]);
    return out;
}

// Reconstructs an open contour as a connected chain with arc endpoints at
// index 0 and n-1. Works on unordered point sets by:
//   1. Building an 8-connectivity lookup from the point set.
//   2. Finding an endpoint: a point with exactly one 8-neighbor in the set.
//      Falls back to contour[0] for closed or degenerate contours.
//   3. Greedily traversing from that endpoint to reconstruct the chain.
// Points not reachable via 8-connectivity are appended at the end so the
// returned vector always contains all input points.
static std::vector<cv::Point> reorderOpenContour(const std::vector<cv::Point>& contour)
{
    const int n = static_cast<int>(contour.size());
    if (n < 2) return contour;

    // Build point set for O(1) neighbour lookup
    std::unordered_map<int, std::unordered_set<int>> ptSet;
    ptSet.reserve(n);
    for (const auto& p : contour)
        ptSet[p.y].insert(p.x);

    auto hasPoint = [&](int x, int y) {
        auto row = ptSet.find(y);
        return row != ptSet.end() && row->second.count(x);
    };

    auto neighborCount = [&](const cv::Point& p) {
        int cnt = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if ((dx || dy) && hasPoint(p.x + dx, p.y + dy)) ++cnt;
        return cnt;
    };

    // Find an endpoint (exactly one 8-neighbour); fall back to contour[0]
    cv::Point start = contour[0];
    for (const auto& p : contour) {
        if (neighborCount(p) == 1) { start = p; break; }
    }

    // Greedy chain traversal
    std::unordered_map<int, std::unordered_set<int>> visited;
    std::vector<cv::Point> ordered;
    ordered.reserve(n);

    cv::Point cur = start;
    while (true) {
        ordered.push_back(cur);
        visited[cur.y].insert(cur.x);

        cv::Point next(-1, -1);
        for (int dy = -1; dy <= 1 && next.x < 0; ++dy)
            for (int dx = -1; dx <= 1 && next.x < 0; ++dx)
                if ((dx || dy) && hasPoint(cur.x + dx, cur.y + dy)) {
                    auto vrow = visited.find(cur.y + dy);
                    if (vrow == visited.end() || !vrow->second.count(cur.x + dx))
                        next = {cur.x + dx, cur.y + dy};
                }

        if (next.x < 0) break;
        cur = next;
    }

    // Append any points not reached via connectivity (gaps in the edge image)
    if (static_cast<int>(ordered.size()) < n) {
        for (const auto& p : contour) {
            auto vrow = visited.find(p.y);
            if (vrow == visited.end() || !vrow->second.count(p.x))
                ordered.push_back(p);
        }
    }

    return ordered;
}

// For a focused-search contour: outer RANSAC tries random endpoint cuts,
// runs ransacRefineEllipse on each trimmed segment, keeps the result with
// the most inliers. Returns an empty vector if no good fit is found.
static std::vector<cv::Point> trimAndRefineContour(
    const std::vector<cv::Point>& contour,
    const FocusedSearchConfig&    focusCfg,
    int                           minPoints)
{
    const int n = static_cast<int>(contour.size());
    constexpr int kMin = 5;
    if (n < kMin) return {};

    std::mt19937 rng(42);
    const int maxCut = static_cast<int>(n * focusCfg.ransac.maxCutFraction);
    std::uniform_int_distribution<int> cutDist(0, std::max(0, maxCut));

    int bestInlierCount = 0;
    std::vector<cv::Point> bestInliers;

    for (int iter = 0; iter < focusCfg.trimIterations; ++iter) {
        // Random endpoint cuts anchored at the first / last contour point
        int cutStart = cutDist(rng);
        int cutEnd   = cutDist(rng);
        if (n - cutStart - cutEnd < kMin) continue;

        std::vector<cv::Point> trimmed(contour.begin() + cutStart,
                                       contour.end()   - cutEnd);

        std::vector<cv::Point> inliers;
        ransacRefineEllipse(trimmed, focusCfg.ransac, inliers);

        if (static_cast<int>(inliers.size()) > bestInlierCount) {
            bestInlierCount = static_cast<int>(inliers.size());
            bestInliers     = inliers;
        }
    }

    if (bestInlierCount >= minPoints) return bestInliers;
    return {};
}

std::vector<std::vector<cv::Point>> detectBallContours(
    const cv::Mat&             frame,
    const BallDetectorConfig&  cfg,
    const FocusedSearchConfig& focusCfg,
    const cv::RotatedRect*     prevEllipse)
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

    // 6. Focused search: re-detect inside a crop around the previous ellipse
    if (focusCfg.enabled && prevEllipse != nullptr) {
        float halfW = prevEllipse->size.width  * 0.5f * static_cast<float>(focusCfg.cropScale);
        float halfH = prevEllipse->size.height * 0.5f * static_cast<float>(focusCfg.cropScale);
        int x0 = std::max(0, static_cast<int>(prevEllipse->center.x - halfW));
        int y0 = std::max(0, static_cast<int>(prevEllipse->center.y - halfH));
        int x1 = std::min(edges.cols, static_cast<int>(prevEllipse->center.x + halfW));
        int y1 = std::min(edges.rows, static_cast<int>(prevEllipse->center.y + halfH));

        if (x1 > x0 && y1 > y0) {
            cv::Rect cropRect(x0, y0, x1 - x0, y1 - y0);
            cv::Mat cropEdges = edges(cropRect).clone();

            std::vector<std::vector<cv::Point>> cropContours;
            //cv::findContours(cropEdges, cropContours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
            cv::findContours(cropEdges, cropContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

            for (auto& fc : cropContours) {
                if (static_cast<int>(fc.size()) < cfg.minContourPoints)
                    continue;

                // Translate crop-local coordinates back to full-image space
                for (auto& p : fc) { p.x += x0; p.y += y0; }

                // Rotate so geometric arc endpoints are at index 0 and n-1
                fc = reorderOpenContour(fc);

                // Outer RANSAC: try random endpoint cuts, run ransacRefineEllipse
                // on each trimmed segment, keep the inlier set with most support
                auto refined = trimAndRefineContour(fc, focusCfg, cfg.minContourPoints);
                if (!refined.empty())
                    result.push_back(std::move(refined));
            }
        }
    }

    // 7. NMS: remove duplicate ellipses introduced by the focused search
    return applyNMS(result, cfg.nmsOverlapFraction);
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

    // Maximum points cuttable from each end
    const int maxCut = static_cast<int>(n * cfg.maxCutFraction);
    std::uniform_int_distribution<int> cutDist(0, std::max(0, maxCut));

    int bestCount = 0;
    cv::RotatedRect bestEl;

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        // Remove random-length segments from both ends of the ordered contour
        int cutStart = cutDist(rng);
        int cutEnd   = cutDist(rng);
        if (n - cutStart - cutEnd < kSample)
            continue;

        std::vector<cv::Point> sample(contour.begin() + cutStart,
                                      contour.end()   - cutEnd);

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
