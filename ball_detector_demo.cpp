#include "ball_detector.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // --- Parse flags ---
    bool focusedMode = false;
    const char* inputPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--focused")
            focusedMode = true;
        else if (inputPath == nullptr)
            inputPath = argv[i];
    }

    // --- Load image or open camera ---
    cv::Mat frame;
    cv::VideoCapture cap;
    bool liveMode = false;
    cv::namedWindow("Ball Detector");

    if (inputPath) {
        frame = cv::imread(inputPath);
        if (frame.empty()) {
            cap.open(inputPath);
            liveMode = cap.isOpened();
            if (!liveMode) {
                std::cerr << "Cannot open: " << inputPath << "\n";
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
    cfg.minRadius        = 15.0;
    cfg.maxRadius        = 200.0;
    cfg.minAxisRatio     = 1.0 / 1.2;  // reject if major > 1.2 * minor
    cfg.cannyLow         = 20.0;
    cfg.cannyHigh        = 60.0;
    cfg.maxFitResidual   = 0.07;
    cfg.minContourPoints = 12;
    cfg.maxInteriorCV    = 0.25;

    FocusedSearchConfig focusCfg;
    focusCfg.enabled = focusedMode;

    // State for focused search: ellipse from the previous frame
    cv::RotatedRect prevEllipse;
    bool hasPrevEllipse = false;

    auto processFrame = [&](const cv::Mat& img) {
        const cv::RotatedRect* prevPtr = hasPrevEllipse ? &prevEllipse : nullptr;
        auto contours = detectBallContours(img, cfg, focusCfg, prevPtr);

        // Update previous ellipse for next frame (use first detection)
        if (!contours.empty() && contours[0].size() >= 5) {
            prevEllipse    = cv::fitEllipseAMS(contours[0]);
            hasPrevEllipse = true;
        } else {
            hasPrevEllipse = false;
        }

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
        const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const bool isVideo = totalFrames > 1; // false for live camera

        int sliderPos = 0;
        if (isVideo)
            cv::createTrackbar("Frame", "Ball Detector", &sliderPos, totalFrames - 1);

        cv::Mat f;
        int lastSetPos = 0;

        while (true) {
            if (isVideo) {
                int pos = cv::getTrackbarPos("Frame", "Ball Detector");
                if (pos != lastSetPos) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, pos);
                    lastSetPos = pos;
                    hasPrevEllipse = false; // temporal continuity broken after seek
                }
            }

            if (!cap.read(f)) break;

            if (isVideo) {
                lastSetPos = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
                cv::setTrackbarPos("Frame", "Ball Detector", lastSetPos);
            }

            cv::Mat vis = processFrame(f);
            cv::imshow("Ball Detector", vis);

            int key = cv::waitKey(30);
            if (key == 27) break;                     // ESC: quit
            if (key == 32 && isVideo) cv::waitKey(0); // Space: pause
        }
    }

    return 0;
}
