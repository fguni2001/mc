

import argparse
import cv2
import numpy as np
from types import SimpleNamespace

from deep_sort_application import gather_sequence_info, create_detections
from deep_sort.iou_matching import iou
from application_util.tracking_visualizer import Displayer

DEFAULT_FRAME_INTERVAL_MS = 20

def render_visualization(
    sequence_path,
    tracking_results_file,
    highlight_false_alarms=False,
    detection_data_file=None,
    frame_interval_ms=None,
    output_video_file=None
):
    """
    Visualize tracking results with optional video output.

    Parameters
    ----------
    sequence_path : str
        Path to the MOTChallenge sequence directory.
    tracking_results_file : str
        Path to the tracking output file (MOTChallenge format).
    highlight_false_alarms : bool
        If True, show false alarms in red.
    detection_data_file : str, optional
        Path to detection data file.
    frame_interval_ms : int, optional
        Milliseconds between frames. Uses seqinfo.ini or default if not provided.
    output_video_file : str, optional
        If given, writes video of visualization to this path.
    """

    # Get sequence info (images, optionally ground truth & detections)
    sequence_info = gather_sequence_info(sequence_path, detection_data_file)

    # Load tracking results robustly
    if tracking_results_file.endswith('.npy'):
        results = np.load(tracking_results_file)
    else:
        with open(tracking_results_file, 'r', encoding='utf-8', errors='ignore') as f:
            results = np.loadtxt(f, delimiter=',')

    if highlight_false_alarms and sequence_info.get("groundtruth") is None:
        raise ValueError("No groundtruth available for false alarm visualization.")

    def draw_frame(display, frame_index):
        print(f"Rendering frame {frame_index}")
        img_path = sequence_info["image_filenames"][frame_index]
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        display.set_image(frame.copy())

        # 1️⃣ Draw raw detections (in red)
        if sequence_info.get("detections") is not None:
            detections = create_detections(sequence_info["detections"], frame_index)
            display.draw_detections(detections)

        # 2️⃣ Draw tracker outputs (rainbow per track_id)
        mask = results[:, 0].astype(int) == frame_index
        track_ids = results[mask, 1].astype(int)
        bboxes    = results[mask, 2:6]

        # Wrap into objects expected by draw_trackers(...)
        tracks = [
            SimpleNamespace(
                track_id=int(tid),
                time_since_update=0,
                is_confirmed=lambda: True,
                to_tlwh=lambda bbox=bbox: bbox
            )
            for tid, bbox in zip(track_ids, bboxes)
        ]
        display.draw_trackers(tracks)

        # 3️⃣ Highlight false alarms if requested
        if highlight_false_alarms:
            gt = sequence_info["groundtruth"]
            mask_gt = gt[:, 0].astype(int) == frame_index
            gt_boxes = gt[mask_gt, 2:6]
            for bbox in bboxes:
                # if no overlap > 0.5, mark as false alarm
                if gt_boxes.size == 0 or iou(bbox, gt_boxes).max() < 0.5:
                    display.viewer.set_color((0, 0, 255))  # BGR red
                    display.viewer.line_width = 4          # thicker stroke
                    display.viewer.draw_rect(*bbox.astype(int))

        return True

    # Determine frame interval
    if frame_interval_ms is None:
        frame_interval_ms = sequence_info.get("update_ms", DEFAULT_FRAME_INTERVAL_MS)

    # Create visualizer
    visualizer = Displayer(sequence_info, frame_interval_ms)
    if output_video_file is not None:
        visualizer.viewer.enable_writer(output_video_file)

    # Run the loop
    visualizer.run(draw_frame)


def run(
    sequence_path,
    tracking_results_file,
    highlight_false_alarms=False,
    detection_data_file=None,
    frame_interval_ms=None,
    output_video_file=None
):
    """
    Alias to render_visualization, used by my_tracker.py.
    """
    render_visualization(
        sequence_path,
        tracking_results_file,
        highlight_false_alarms,
        detection_data_file,
        frame_interval_ms,
        output_video_file
    )


def get_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tracking Output Rendering")
    parser.add_argument(
        "--sequence_dir", required=True,
        help="Path to the MOTChallenge sequence directory."
    )
    parser.add_argument(
        "--result_file", required=True,
        help="Tracking output file (MOTChallenge format)."
    )
    parser.add_argument(
        "--detection_file", default=None,
        help="Path to custom detection data file (optional)."
    )
    parser.add_argument(
        "--update_ms", default=None, type=int,
        help="Milliseconds between frames (default from seqinfo.ini)."
    )
    parser.add_argument(
        "--output_file", default=None,
        help="Optional output video filename."
    )
    parser.add_argument(
        "--show_false_alarms", action="store_true",
        help="Show false alarms as red bounding boxes."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    render_visualization(
        sequence_path=args.sequence_dir,
        tracking_results_file=args.result_file,
        highlight_false_alarms=args.show_false_alarms,
        detection_data_file=args.detection_file,
        frame_interval_ms=args.update_ms,
        output_video_file=args.output_file
    )
