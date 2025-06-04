
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2

# DeepSORT internals
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

#Utility modules
from application_utils.detection_filter import nms_boxes
from application_utils.tracking_visualizer import Displayer

# ————————————————
# Helpers to load a sequence and its detections
# ————————————————

def gather_sequence_info(sequence_dir, detection_file):
    """Gather image paths, detections, groundtruth, frame‐range & frameRate."""
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
    }
    gt_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None and os.path.exists(detection_file):
        detections = np.load(detection_file)
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
        min_frame = int(detections[:,0].min())
        max_frame = int(detections[:,0].max())

    seqinfo = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(seqinfo):
        with open(seqinfo, "r") as f:
            lines = [l.split('=') for l in f.read().splitlines()[1:]]
            info = dict(s for s in lines if isinstance(s, list) and len(s)==2)
        update_ms = 1000 / int(info["frameRate"])
    else:
        update_ms = None

    return {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame,
        "max_frame_idx": max_frame,
        "update_ms": update_ms
    }

def create_detections(det_mat, frame_idx, min_height=0):
    """Turn one row of your detection matrix into a Detection object."""
    frame_ids = det_mat[:,0].astype(np.int64)
    mask = frame_ids == frame_idx
    dets = []
    for row in det_mat[mask]:
        bbox = row[2:6]
        conf = row[6]
        feat = row[10:]
        if bbox[3] < min_height:
            continue
        dets.append(Detection(bbox, conf, feat))
    return dets

# ————————————
# Run DeepSORT 
# ————————————

def run_tracker(
    sequence_dir,
    detection_file,
    output_file,
    min_confidence=0.3,
    nms_max_overlap=1.0,
    min_detection_height=0,
    max_cosine_distance=0.2,
    nn_budget=None,
    display=False
):
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)
    results = []
    def frame_callback(vis, frame_idx):
        dets = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        dets = [d for d in dets if d.confidence >= min_confidence]
        boxes  = np.array([d.tlwh for d in dets])
        scores = np.array([d.confidence for d in dets])
        keep   = nms_boxes(boxes, nms_max_overlap, scores)
        dets   = [dets[i] for i in keep]
        tracker.predict()
        tracker.update(dets)
        if display:
            img = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR
            )
            vis.set_image(img.copy())
            vis.draw_detections(dets)
            vis.draw_trackers(tracker.tracks)
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update > 0:
                continue
            x, y, w, h = tr.to_tlwh()
            results.append([frame_idx, tr.track_id, x, y, w, h])
    if display:
        vis = Displayer(seq_info, seq_info["update_ms"] or 20)
    else:
        from application_util.tracking_visualizer import DummyVisualizer
        vis = DummyVisualizer(seq_info)

    vis.run(frame_callback)

    # write MOT‐format output: frame,id,x,y,w,h,1,-1,-1,-1
    with open(output_file, 'w') as f:
        for fr, tid, x, y, w, h in results:
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

# ————————————————
# CLI & orchestration
# ————————————————

def parse_args():
    p = argparse.ArgumentParser(description="PeopleTracker (Neti’s DeepSORT)")
    p.add_argument('--dataset_path',    required=True)
    p.add_argument('--detections_path', required=True)
    p.add_argument('--output_path',     required=True)
    p.add_argument('--display',         action='store_true')
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    det_files = ( [args.detections_path]
                  if not os.path.isdir(args.detections_path)
                  else sorted(os.listdir(args.detections_path)) )

    for det in tqdm(det_files, desc="Sequences"):
        det_file = det if os.path.isabs(det) else os.path.join(args.detections_path, det)
        seq_name = os.path.splitext(os.path.basename(det_file))[0]
        out_txt  = os.path.join(args.output_path, f"{seq_name}.txt")
        out_vid  = os.path.join(args.output_path, f"{seq_name}.avi")

        # Run tracker → produces out_txt
        run_tracker(
            args.dataset_path,
            det_file,
            out_txt,
            display=args.display
        )

        # Visualize those results
        import result_renderer
        result_renderer.run(
            sequence_path=args.dataset_path,
            tracking_results_file=out_txt,
            highlight_false_alarms=False,
            detection_data_file=det_file,
            frame_interval_ms=None,
            output_video_file=out_vid
        )

if __name__ == "__main__":
    main()
