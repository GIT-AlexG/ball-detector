#include "ball_detector.hpp"
#include <iostream>
#include <limits>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[]) {
    // --- Parse flags ---
    bool focusedMode  = false;
    bool templateMode = false;
    bool saveMode     = false;
    const char* inputPath = nullptr;
    int startFrame = 0;
    for (int i = 1; i < argc; ++i) {
        if      (std::string(argv[i]) == "--focused")  focusedMode  = true;
        else if (std::string(argv[i]) == "--template") templateMode = true;
        else if (std::string(argv[i]) == "--save")     saveMode     = true;
        else if (std::string(argv[i]) == "--start-frame" && i + 1 < argc)
            startFrame = std::atoi(argv[++i]);
        else if (inputPath == nullptr)                 inputPath    = argv[i];
    }

    // --- Output directory for saved frames (named by current time) ---
    std::filesystem::path saveDir;
    if (saveMode) {
        std::time_t t  = std::time(nullptr);
        std::tm     lt = *std::localtime(&t);
        std::ostringstream name;
        name << std::put_time(&lt, "%Y-%m-%d_%H-%M-%S");
        saveDir = std::filesystem::current_path() / name.str();
        std::filesystem::create_directories(saveDir);
        std::cout << "Saving frames to: " << saveDir.string() << "\n";
    }
    int saveCounter = 0;

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

    // --- Adaptive axis-ratio gating ---
    // Once a ball has been tracked for a while we know its axis ratio fairly
    // well; it may drift slowly but never jumps. So we progressively tighten the
    // allowed minAxisRatio around the learned value, rejecting spurious elongated
    // fits. After a sustained miss the tolerance is reset to the initial value.
    const double initialMinAxisRatio = cfg.minAxisRatio;
    constexpr int    kWarmupFrames = 5;     // hits before tightening starts
    constexpr int    kResetMisses  = 5;     // misses before resetting tolerance
    constexpr double kLooseTol     = 0.15;  // tolerance just after warmup
    constexpr double kTightTol     = 0.05;  // tolerance floor for a long track
    constexpr double kTolStep      = 0.02;  // tolerance shrink per extra hit
    double trackedRatio = 0.0;              // running estimate of minor/major
    int    trackStreak  = 0;                // consecutive successful detections
    int    missStreak   = 0;                // consecutive frames without a ball

    auto processFrame = [&](const cv::Mat& img) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Tighten the allowed axis ratio once the track is established. The
        // tolerance shrinks the longer the ball has been followed; below the
        // warmup count the initial (loose) value is used.
        if (trackStreak >= kWarmupFrames && trackedRatio > 0.0) {
            double tol = std::max(kTightTol,
                                  kLooseTol - (trackStreak - kWarmupFrames) * kTolStep);
            cfg.minAxisRatio = std::max(initialMinAxisRatio, trackedRatio - tol);
        } else {
            cfg.minAxisRatio = initialMinAxisRatio;
        }

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
        cv::Mat focusCanny;
        auto contours = detectBallContours(roiFrame, cfg, focusCfg, prevPtr,
                                           focusCfg.enabled ? &focusCanny : nullptr);

        // Translate contour points back to full-image coordinates
        for (auto& c : contours)
            for (auto& p : c)
                p += cv::Point(searchROI.x, searchROI.y);

        // --- Select the best contour ---
        // When a template match exists, pick the contour whose ellipse center is
        // closest to the match position (avoids snapping onto larger spurious
        // contours). Otherwise fall back to contours[0] (largest after NMS).
        int bestIdx = -1;
        {
            cv::Point2f anchor = (tmplMatchCenter.x >= 0)
                                     ? tmplMatchCenter
                                     : (hasPrevEllipse ? prevEllipse.center
                                                       : cv::Point2f(-1.f, -1.f));
            float bestDistSq = std::numeric_limits<float>::max();
            for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
                if (contours[i].size() < 5) continue;
                if (anchor.x < 0) { bestIdx = i; break; } // no anchor → take first valid
                cv::RotatedRect el = cv::fitEllipseAMS(contours[i]);
                float dx = el.center.x - anchor.x;
                float dy = el.center.y - anchor.y;
                float dSq = dx * dx + dy * dy;
                if (dSq < bestDistSq) { bestDistSq = dSq; bestIdx = i; }
            }
        }

        // --- Create / update template from the best contour ---
        if (tmplCfg.enabled && bestIdx >= 0) {
            cv::RotatedRect bestEl   = cv::fitEllipseAMS(contours[bestIdx]);
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
                    // Weighted update: 20 % new patch, 80 % existing template
                    cv::Mat newPatchResized;
                    cv::resize(newPatch, newPatchResized, ellipseTemplate.size());
                    cv::addWeighted(newPatchResized, 0.2,
                                    ellipseTemplate,  0.8,
                                    0.0, ellipseTemplate);
                }
            }
        }

        // --- Update prevEllipse for next frame ---
        // If a contour was found, update with the precise fit of the best one.
        // Otherwise keep all parameters from the previous frame unchanged
        // so the next frame's search stays anchored at the last known position.
        if (bestIdx >= 0) {
            prevEllipse    = cv::fitEllipseAMS(contours[bestIdx]);
            hasPrevEllipse = true;
        } else if (tmplMatchCenter.x >= 0) {
            hasPrevEllipse = true; // prevEllipse.center already updated by template match
        }
        // else: hasPrevEllipse and prevEllipse remain as-is

        // --- Update adaptive axis-ratio tracking ---
        if (bestIdx >= 0) {
            double major = std::max(prevEllipse.size.width, prevEllipse.size.height) * 0.5;
            double minor = std::min(prevEllipse.size.width, prevEllipse.size.height) * 0.5;
            double ratio = (major > 0.0) ? (minor / major) : 0.0;
            // Running average so the learned ratio drifts slowly with the ball.
            trackedRatio = (trackStreak == 0) ? ratio
                                              : 0.8 * trackedRatio + 0.2 * ratio;
            ++trackStreak;
            missStreak = 0;
        } else if (++missStreak >= kResetMisses) {
            // Lost the ball for a while → forget the learned ratio.
            trackStreak  = 0;
            trackedRatio = 0.0;
        }

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

            float major = std::max(elRansac.size.width, elRansac.size.height) * 0.5f;
            float minor = std::min(elRansac.size.width, elRansac.size.height) * 0.5f;

            // The RANSAC fit uses only a subset of points and is not run through
            // detectBallContours' shape filters, so it can be far more elongated
            // than the accepted contour. Reject it here with the same axis-ratio
            // gate before drawing, otherwise stray elongated ellipses appear.
            double ransacRatio = (major > 0.f) ? (minor / major) : 0.0;
            if (ransacRatio < cfg.minAxisRatio)
                continue;

            cv::ellipse(vis, elRansac, {255, 255, 0}, 2);
            for (const auto& p : inliers)
                cv::circle(vis, p, 2, {0, 255, 255}, -1);
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

        // --- Inset overlays in the top-left corner ---
        // Draw the focused-search Canny crop and the current template directly
        // into vis (side by side) instead of using separate windows.
        {
            int insetX = 5;
            const int insetY = 5;
            auto blit = [&](const cv::Mat& src) {
                if (src.empty()) return;
                cv::Mat bgr;
                if (src.channels() == 1) cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
                else                     bgr = src;
                int w = std::min(bgr.cols, vis.cols - insetX);
                int h = std::min(bgr.rows, vis.rows - insetY);
                if (w <= 0 || h <= 0) return;
                cv::Rect dst(insetX, insetY, w, h);
                bgr(cv::Rect(0, 0, w, h)).copyTo(vis(dst));
                cv::rectangle(vis, dst, {255, 255, 255}, 1);
                insetX += w + 5;  // next inset goes to the right
            };

            if (focusCfg.enabled) blit(focusCanny);
            if (tmplCfg.enabled && hasTemplate) blit(ellipseTemplate);

            // Top-right: crop around the final ball with its fitted ellipse drawn in.
            if (bestIdx >= 0) {
                cv::RotatedRect finalEl = cv::fitEllipseAMS(contours[bestIdx]);
                cv::Rect r = finalEl.boundingRect();
                // Pad the crop a little so the whole ellipse is visible
                int pad = 8;
                r.x -= pad; r.y -= pad; r.width += 2 * pad; r.height += 2 * pad;
                r &= cv::Rect(0, 0, img.cols, img.rows);
                if (r.width > 0 && r.height > 0) {
                    cv::Mat crop = img(r).clone();
                    // Shift the ellipse into crop-local coordinates and draw it
                    cv::RotatedRect localEl = finalEl;
                    localEl.center.x -= r.x;
                    localEl.center.y -= r.y;
                    cv::ellipse(crop, localEl, {0, 0, 255}, 2);

                    int w = std::min(crop.cols, vis.cols - 5);
                    int h = std::min(crop.rows, vis.rows - 5);
                    if (w > 0 && h > 0) {
                        cv::Rect dst(vis.cols - w - 5, 5, w, h);
                        crop(cv::Rect(0, 0, w, h)).copyTo(vis(dst));
                        cv::rectangle(vis, dst, {255, 255, 255}, 1);
                    }

                    // Second crop below: same region, but with all the contour
                    // points that were used to fit the ellipse drawn in.
                    cv::Mat cropPts = img(r).clone();
                    for (const auto& p : contours[bestIdx])
                        cv::circle(cropPts, cv::Point(p.x - r.x, p.y - r.y),
                                   1, {0, 255, 255}, -1);

                    int y2 = 5 + h + 5;
                    int w2 = std::min(cropPts.cols, vis.cols - 5);
                    int h2 = std::min(cropPts.rows, vis.rows - y2);
                    if (w2 > 0 && h2 > 0) {
                        cv::Rect dst2(vis.cols - w2 - 5, y2, w2, h2);
                        cropPts(cv::Rect(0, 0, w2, h2)).copyTo(vis(dst2));
                        cv::rectangle(vis, dst2, {255, 255, 255}, 1);
                    }
                }
            }
        }

        std::cout << "Detected " << contours.size() << " contour(s)";
        if (tmplCfg.enabled && hasTemplate)
            std::cout << "  |  TM score: " << cv::format("%.3f", tmplMatchScore)
                      << (tmplMatchCenter.x >= 0 ? " [match]" : " [miss]");
        std::cout << "\n";
        return vis;
    };

    if (!liveMode) {
        cv::Mat vis = processFrame(frame);
        cv::imshow("Ball Detector", vis);
        if (saveMode)
            cv::imwrite((saveDir / "frame_0000.png").string(), vis);
        cv::waitKey(0);
    } else {
        const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        const bool isVideo = totalFrames > 1;

        int sliderPos = 0;
        if (isVideo)
            cv::createTrackbar("Frame", "Ball Detector", &sliderPos, totalFrames - 1);

        cv::Mat f;
        int  lastSetPos  = 0;
        bool stepMode    = true;   // P toggles; Space advances one frame
        bool stepPending = false;

        // Seek to start frame if requested
        if (isVideo && startFrame > 0) {
            int sf = std::min(startFrame, totalFrames - 1);
            cap.set(cv::CAP_PROP_POS_FRAMES, sf);
            lastSetPos = sf;
            cv::setTrackbarPos("Frame", "Ball Detector", sf);
        }

        while (true) {
            if (isVideo) {
                int pos = cv::getTrackbarPos("Frame", "Ball Detector");
                if (pos != lastSetPos) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, pos);
                    lastSetPos     = pos;
                    hasPrevEllipse = false;
                    stepPending    = true;  // show the seeked frame immediately
                }
            }

            // In step mode only advance when Space was pressed (or a seek happened)
            bool doRead = !stepMode || stepPending;
            stepPending = false;

            if (doRead) {
                if (!cap.read(f)) break;

                if (isVideo) {
                    lastSetPos = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
                    cv::setTrackbarPos("Frame", "Ball Detector", lastSetPos);
                }

                cv::Mat vis = processFrame(f);
                cv::imshow("Ball Detector", vis);

                if (saveMode) {
                    int frameIdx = isVideo ? lastSetPos : saveCounter++;
                    std::ostringstream fn;
                    fn << "frame_" << std::setw(5) << std::setfill('0')
                       << frameIdx << ".png";
                    cv::imwrite((saveDir / fn.str()).string(), vis);
                }
            }

            int key = cv::waitKey(stepMode ? 50 : 30);
            if (key == 27)  break;                      // Esc — quit
            if (key == 'p' || key == 'P') {             // P — toggle step/continuous
                stepMode    = !stepMode;
                stepPending = false;
                std::cout << (stepMode ? "[step mode]\n" : "[continuous mode]\n");
            }
            if (key == 32 && isVideo && stepMode)       // Space — advance one frame
                stepPending = true;
        }
    }

    return 0;
}
