import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from tqdm import tqdm
import numpy as np
import cv2

# DeepSORT internals
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from yolo_detection import YOLOv3Detector
from tools.extract_features import FeatureExtractor  # <--- FIXED path

# Utility modules
from application_utils.detection_filter import nms_boxes
from application_utils.tracking_visualizer import Displayer, DummyVisualizer

def gather_sequence_info(sequence_dir):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png'))
    }
    gt_file = os.path.join(sequence_dir, "gt/gt.txt")
    groundtruth = None
    if os.path.exists(gt_file):
        groundtruth = np.loadtxt(gt_file, delimiter=',')

    if image_filenames:
        sample = next(iter(image_filenames.values()))
        img = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
        image_size = img.shape
        min_frame = min(image_filenames)
        max_frame = max(image_filenames)
    else:
        image_size = None
        min_frame = 0
        max_frame = 0

    seqinfo = os.path.join(sequence_dir, "seqinfo.ini")
    update_ms = None
    if os.path.exists(seqinfo):
        with open(seqinfo, "r") as f:
            lines = [l.split('=') for l in f.read().splitlines()[1:]]
            info = dict(s for s in lines if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info["frameRate"])

    return {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame,
        "max_frame_idx": max_frame,
        "update_ms": update_ms
    }

def run_tracker(
    sequence_dir,
    output_file,
    min_confidence=0.3,
    nms_max_overlap=1.0,
    min_detection_height=0,
    max_cosine_distance=0.2,
    nn_budget=None,
    display=False,
    use_yolo=True,
    yolo_weights='weights/yolov3.pt',
    feature_model_path='networks/mars-small128.pb',
    max_frames=None
):
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)
    results = []

    yolo = YOLOv3Detector(weights_path=yolo_weights, device='cpu')
    feature_extractor = FeatureExtractor(model_path=feature_model_path)

    def frame_callback(vis, frame_idx):
        if max_frames is not None and (frame_idx - seq_info["min_frame_idx"] + 1) > max_frames:
            if hasattr(vis, 'stop'):
                vis.stop()  # For SimpleImageDisplayer or custom viewers
            return

        img_path = seq_info["image_filenames"].get(frame_idx, None)
        frame = cv2.imread(img_path) if img_path else None
        if frame is None:
            print(f"⚠️ Could not read image at frame {frame_idx}: {img_path}")
            return

        dets = []
        yolo_dets = yolo.detect(frame)
        h_img, w_img = frame.shape[:2]

        for det in yolo_dets:
            x, y, w, h, conf = det
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                print(f"⚠️ Skipping invalid detection: {det}")
                continue

            x = int(np.clip(x, 0, w_img - 1))
            y = int(np.clip(y, 0, h_img - 1))
            w = int(np.clip(w, 1, w_img - x))
            h = int(np.clip(h, 1, h_img - y))

            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                print(f"⚠️ Skipping empty crop for detection: {det}")
                continue

            crop = cv2.resize(crop, (64, 128))
            feature = feature_extractor.encode_patches(np.array([crop]))[0]
            dets.append(Detection(np.array([x, y, w, h]), conf, feature))

        # Filter and track
        dets = [d for d in dets if d.confidence >= min_confidence]
        if dets:
            boxes = np.array([d.tlwh for d in dets])
            scores = np.array([d.confidence for d in dets])
            keep = nms_boxes(boxes, nms_max_overlap, scores)
            dets = [dets[i] for i in keep]

        tracker.predict()
        tracker.update(dets)

        if display:
            img = frame.copy()
            vis.set_image(img)
            vis.draw_detections(dets)
            vis.draw_trackers(tracker.tracks)

        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update > 0:
                continue
            x, y, w, h = tr.to_tlwh()
            results.append([frame_idx, tr.track_id, x, y, w, h])

    vis = Displayer(seq_info, seq_info["update_ms"] or 20) if display else DummyVisualizer(seq_info)
    vis.run(frame_callback)

    with open(output_file, 'w') as f:
        for fr, tid, x, y, w, h in results:
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

def parse_args():
    p = argparse.ArgumentParser(description="PeopleTracker (DeepSORT+YOLO)")
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--display', action='store_true')
    p.add_argument('--yolo_weights', default='weights/yolov3.pt')
    p.add_argument('--feature_model_path', default='networks/mars-small128.pb')
    p.add_argument('--max_frames', type=int, default=None, help="Maximum number of frames to process (for quick testing)")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Expect dataset_path to point to a sequence folder, e.g. .../MOT17-02-SDP
    seq_name = os.path.basename(os.path.normpath(args.dataset_path))
    out_txt = os.path.join(args.output_path, f"{seq_name}.txt")
    out_vid = os.path.join(args.output_path, f"{seq_name}.avi")

    run_tracker(
        sequence_dir=args.dataset_path,
        output_file=out_txt,
        display=args.display,
        use_yolo=True,
        yolo_weights=args.yolo_weights,
        feature_model_path=args.feature_model_path,
        max_frames=args.max_frames
    )

    # Optional: render visualization video
    import result_renderer
    result_renderer.run(
        sequence_path=args.dataset_path,
        tracking_results_file=out_txt,
        highlight_false_alarms=False,
        detection_data_file=None,
        frame_interval_ms=None,
        output_video_file=out_vid
    )

if __name__ == "__main__":
    main()
