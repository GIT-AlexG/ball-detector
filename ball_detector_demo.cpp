#include "ball_detector.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // --- Load image or open camera ---
    cv::Mat frame;
    cv::VideoCapture cap;
    bool liveMode = false;

    if (argc > 1) {
        frame = cv::imread(argv[1]);
        //frame = frame(cv::Rect(0, 0, 1498, 1200));
        if (frame.empty()) {
            // Try as video
            cap.open(argv[1]);
            liveMode = cap.isOpened();
            if (!liveMode) {
                std::cerr << "Cannot open: " << argv[1] << "\n";
                return 1;
            }
        }
    } else {
        cap.open(0);
        liveMode = cap.isOpened();
        if (!liveMode) {
            std::cerr << "No camera found\n";
            return 1;
        }
    }

    // --- Configuration ---
    BallDetectorConfig cfg;
    RansacConfig ransacCfg;
    cfg.minRadius       = 15.0;
    cfg.maxRadius       = 200.0;
    cfg.minAxisRatio    = 0.45;
    cfg.cannyLow        = 20.0;
    cfg.cannyHigh       = 60.0;
    cfg.maxFitResidual  = 0.07;
    cfg.minContourPoints = 12;
    cfg.maxInteriorCV   = 0.25;

    auto processFrame = [&](const cv::Mat& img) {
        auto contours = detectBallContours(img, cfg);

        cv::Mat vis = img.clone();
        cv::drawContours(vis, contours, -1, {0, 255, 0}, 2);

        for (const auto& c : contours) {
            if (c.size() < 5) continue;

            // Initial ellipse fit (red)
            cv::RotatedRect el = cv::fitEllipseAMS(c);
            cv::ellipse(vis, el, {0, 0, 255}, 2);

            // RANSAC-refined ellipse (cyan), inliers as small dots (yellow)
            std::vector<cv::Point> inliers;
            cv::RotatedRect elRansac = ransacRefineEllipse(c, ransacCfg, inliers);
            cv::ellipse(vis, elRansac, {255, 255, 0}, 2);
            for (const auto& p : inliers)
                cv::circle(vis, p, 2, {0, 255, 255}, -1);

            // Label: axis ratio / inlier count
            float major = std::max(elRansac.size.width, elRansac.size.height) * 0.5f;
            float minor = std::min(elRansac.size.width, elRansac.size.height) * 0.5f;
            std::string label = cv::format("%.2f  %d/%d",
                                           minor / major,
                                           static_cast<int>(inliers.size()),
                                           static_cast<int>(c.size()));
            cv::putText(vis, label,
                        cv::Point(static_cast<int>(elRansac.center.x),
                                  static_cast<int>(elRansac.center.y)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
        }

        std::cout << "Detected " << contours.size() << " contour(s)\n";
        return vis;
    };

    if (!liveMode) {
        cv::Mat vis = processFrame(frame);
        cv::imshow("Ball Detector", vis);
        cv::waitKey(0);
    } else {
        cv::Mat f;
        while (cap.read(f)) {
            cv::Mat vis = processFrame(f);
            cv::imshow("Ball Detector", vis);
            if (cv::waitKey(1) == 27) break;
        }
    }

    return 0;
}
