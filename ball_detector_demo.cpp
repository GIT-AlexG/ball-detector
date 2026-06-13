#include "ball_detector.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    // --- Parse flags ---
    bool focusedMode  = false;
    bool templateMode = false;
    const char* inputPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if      (std::string(argv[i]) == "--focused")  focusedMode  = true;
        else if (std::string(argv[i]) == "--template") templateMode = true;
        else if (inputPath == nullptr)                 inputPath    = argv[i];
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
    RansacConfig       ransacCfg;
    cfg.minRadius        = 15.0;
    cfg.maxRadius        = 200.0;
    cfg.minAxisRatio     = 1.0 / 1.3;  // reject if major > 1.3 * minor
    cfg.cannyLow         = 20.0;
    cfg.cannyHigh        = 60.0;
    cfg.maxFitResidual   = 0.07;
    cfg.minContourPoints = 12;
    cfg.maxInteriorCV    = 0.25;

    FocusedSearchConfig focusCfg;
    focusCfg.enabled = focusedMode;

    TemplateConfig tmplCfg;
    tmplCfg.enabled = templateMode;

    // --- Persistent state across frames ---
    cv::RotatedRect prevEllipse;
    bool            hasPrevEllipse = false;

    cv::Mat ellipseTemplate;
    bool    hasTemplate = false;

    auto processFrame = [&](const cv::Mat& img) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // --- Template matching: update prevEllipse before detection ---
        cv::Point2f tmplMatchCenter(-1.f, -1.f);
        float       tmplMatchScore = 0.f;

        if (tmplCfg.enabled && hasTemplate && hasPrevEllipse) {
            float halfW = ellipseTemplate.cols * 0.5f * static_cast<float>(tmplCfg.searchScale);
            float halfH = ellipseTemplate.rows * 0.5f * static_cast<float>(tmplCfg.searchScale);
            int sx0 = std::max(0,         static_cast<int>(prevEllipse.center.x - halfW));
            int sy0 = std::max(0,         static_cast<int>(prevEllipse.center.y - halfH));
            int sx1 = std::min(gray.cols, static_cast<int>(prevEllipse.center.x + halfW));
            int sy1 = std::min(gray.rows, static_cast<int>(prevEllipse.center.y + halfH));

            if (sx1 - sx0 >= ellipseTemplate.cols && sy1 - sy0 >= ellipseTemplate.rows) {
                cv::Mat searchRegion = gray(cv::Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
                cv::Mat matchResult;
                cv::matchTemplate(searchRegion, ellipseTemplate, matchResult, cv::TM_CCOEFF_NORMED);

                double    maxVal;
                cv::Point maxLoc;
                cv::minMaxLoc(matchResult, nullptr, &maxVal, nullptr, &maxLoc);
                tmplMatchScore = static_cast<float>(maxVal);

                if (maxVal >= tmplCfg.matchThreshold) {
                    prevEllipse.center.x = sx0 + maxLoc.x + ellipseTemplate.cols * 0.5f;
                    prevEllipse.center.y = sy0 + maxLoc.y + ellipseTemplate.rows * 0.5f;
                    tmplMatchCenter = prevEllipse.center;
                }
            }
        }

        // --- Restrict detection to ROI when template match succeeded ---
        // Full frame by default; shrinks to ellipse neighbourhood on match.
        cv::Rect searchROI(0, 0, img.cols, img.rows);
        if (tmplCfg.enabled && tmplMatchCenter.x >= 0) {
            float halfW = prevEllipse.size.width  * 0.5f * static_cast<float>(tmplCfg.roiScale);
            float halfH = prevEllipse.size.height * 0.5f * static_cast<float>(tmplCfg.roiScale);
            int rx0 = std::max(0,         static_cast<int>(prevEllipse.center.x - halfW));
            int ry0 = std::max(0,         static_cast<int>(prevEllipse.center.y - halfH));
            int rx1 = std::min(img.cols,  static_cast<int>(prevEllipse.center.x + halfW));
            int ry1 = std::min(img.rows,  static_cast<int>(prevEllipse.center.y + halfH));
            if (rx1 > rx0 && ry1 > ry0)
                searchROI = cv::Rect(rx0, ry0, rx1 - rx0, ry1 - ry0);
        }

        // Translate prevEllipse to ROI-local coords for focused search inside crop
        cv::RotatedRect prevEllipseLocal = prevEllipse;
        prevEllipseLocal.center.x -= searchROI.x;
        prevEllipseLocal.center.y -= searchROI.y;
        const cv::RotatedRect* prevPtr =
            hasPrevEllipse ? &prevEllipseLocal : nullptr;

        // Run contour detection on the (possibly cropped) frame
        cv::Mat roiFrame = img(searchROI);
        auto contours = detectBallContours(roiFrame, cfg, focusCfg, prevPtr);

        // Translate contour points back to full-image coordinates
        for (auto& c : contours)
            for (auto& p : c)
                p += cv::Point(searchROI.x, searchROI.y);

        // --- Create / update template from detection ---
        if (tmplCfg.enabled && !contours.empty() && contours[0].size() >= 5) {
            cv::RotatedRect bestEl   = cv::fitEllipseAMS(contours[0]);
            cv::Rect        tmplRect = bestEl.boundingRect();
            tmplRect &= cv::Rect(0, 0, gray.cols, gray.rows);
            if (tmplRect.width > 0 && tmplRect.height > 0) {
                cv::Mat newPatch = gray(tmplRect);
                if (!hasTemplate) {
                    ellipseTemplate = newPatch.clone();
                    hasTemplate     = true;
                    std::cout << "Template created: "
                              << tmplRect.width << "x" << tmplRect.height << " px\n";
                } else {
                    // Weighted update: 70 % new patch, 30 % existing template
                    cv::Mat newPatchResized;
                    cv::resize(newPatch, newPatchResized, ellipseTemplate.size());
                    cv::addWeighted(newPatchResized, 0.7,
                                    ellipseTemplate,  0.3,
                                    0.0, ellipseTemplate);
                }
            }
        }

        // Show current template in a separate window
        if (tmplCfg.enabled && hasTemplate)
            cv::imshow("Template", ellipseTemplate);

        // --- Update prevEllipse for next frame ---
        // If a contour was found, update with the precise fit.
        // Otherwise keep all parameters from the previous frame unchanged
        // so the next frame's search stays anchored at the last known position.
        if (!contours.empty() && contours[0].size() >= 5) {
            prevEllipse    = cv::fitEllipseAMS(contours[0]);
            hasPrevEllipse = true;
        } else if (tmplMatchCenter.x >= 0) {
            hasPrevEllipse = true; // prevEllipse.center already updated by template match
        }
        // else: hasPrevEllipse and prevEllipse remain as-is

        // --- Visualisation ---
        cv::Mat vis = img.clone();

        // Active search ROI (blue rectangle, only when restricted)
        if (tmplCfg.enabled && searchROI != cv::Rect(0, 0, img.cols, img.rows))
            cv::rectangle(vis, searchROI, {255, 80, 0}, 1);

        cv::drawContours(vis, contours, -1, {0, 255, 0}, 2);

        for (const auto& c : contours) {
            if (c.size() < 5) continue;

            cv::RotatedRect el = cv::fitEllipseAMS(c);
            cv::ellipse(vis, el, {0, 0, 255}, 2);

            std::vector<cv::Point> inliers;
            cv::RotatedRect elRansac = ransacRefineEllipse(c, ransacCfg, inliers);
            cv::ellipse(vis, elRansac, {255, 255, 0}, 2);
            for (const auto& p : inliers)
                cv::circle(vis, p, 2, {0, 255, 255}, -1);

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

        // Template match indicator (orange circle + score)
        if (tmplMatchCenter.x >= 0) {
            cv::Point center(static_cast<int>(tmplMatchCenter.x),
                             static_cast<int>(tmplMatchCenter.y));
            cv::circle(vis, center,
                       std::max(ellipseTemplate.cols, ellipseTemplate.rows) / 2,
                       {0, 128, 255}, 2);
            cv::putText(vis, cv::format("TM %.2f", tmplMatchScore),
                        center + cv::Point(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 128, 255}, 1);
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
        const bool isVideo = totalFrames > 1;

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
                    lastSetPos     = pos;
                    hasPrevEllipse = false;
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
            if (key == 27) break;
            if (key == 32 && isVideo) cv::waitKey(0);
        }
    }

    return 0;
}
