import argparse
import os
import sys

import cv2 as cv

import behaviour
import detection
import features
import report
import tracking
import visualization
from logger import logger
from utils import config

# ── Default video path (relative to project root) ─────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_DEFAULT_VIDEO = os.path.join(_ROOT_DIR, "test_footage", "drive_test.mp4")
_REPORT_PATH = os.path.join(_ROOT_DIR, "report.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Driver Behaviour Analysis — real-time vehicle monitoring"
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Video file path to analyse, OR camera index (e.g. 0 for webcam). "
            f"Defaults to '{_DEFAULT_VIDEO}' if the file exists, otherwise falls back to webcam (0)."
        ),
    )
    return parser.parse_args()


def resolve_source(source_arg):
    """Return the right cv.VideoCapture source: file path or camera int."""
    if source_arg is not None:
        # User explicitly provided something
        if source_arg.isdigit():
            return int(source_arg)          # camera index
        if os.path.isfile(source_arg):
            return source_arg
        logger.error(f"Provided --source '{source_arg}' not found.")
        sys.exit(1)

    # No argument given → try default video, else webcam
    if os.path.isfile(_DEFAULT_VIDEO):
        logger.info(f"Using default video: {_DEFAULT_VIDEO}")
        return _DEFAULT_VIDEO

    logger.warning(
        f"Default video not found at '{_DEFAULT_VIDEO}'. Falling back to webcam (index 0)."
    )
    return 0


def captureVideo(source):
    latest_results = {}

    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    if isinstance(source, str):
        # For video files, honour the config FPS; for webcam let the driver decide
        cap.set(cv.CAP_PROP_FPS, config.FPS)

    actual_fps = cap.get(cv.CAP_PROP_FPS)
    logger.info(f"Source opened — FPS: {actual_fps:.1f} | Source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("End of stream or cannot receive frame. Stopping.")
            break

        detections = detection.predict(frame)
        tracks = tracking.update_tracks(detections)
        feature = features.extract_features(tracks)
        results = behaviour.analyze_behaviour(feature)

        if isinstance(results, list):
            for r in results:
                latest_results[r["id"]] = r

        logger.debug(f"results: {results}")

        drawn_frame = visualization.draw(frame, tracks, results)
        cv.imshow("Driver Behaviour Analysis  |  Press Q to quit", drawn_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            logger.info("User requested quit.")
            break

    report.generate_report(latest_results, filename=_REPORT_PATH)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    source = resolve_source(args.source)
    captureVideo(source)
