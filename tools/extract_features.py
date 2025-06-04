import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def extract_image_patch(image, bbox, patch_size=(128, 64)):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    target_aspect = float(patch_size[1]) / patch_size[0]
    new_w = target_aspect * h
    x_center = x + w / 2.0
    x1 = int(x_center - new_w / 2.0)
    y1 = int(y)
    x2 = int(x_center + new_w / 2.0)
    y2 = int(y + h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    patch = image[y1:y2, x1:x2]
    patch = cv2.resize(patch, (patch_size[1], patch_size[0]))
    return patch

class FeatureExtractor:
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph)
            self.input_var = self.graph.get_tensor_by_name('images:0')
            self.output_var = self.graph.get_tensor_by_name('features:0')

    def encode_patches(self, patches):
        patches = patches.astype(np.float32) / 255.
        return self.sess.run(self.output_var, feed_dict={self.input_var: patches})

    def extract_features(self, image, bboxes):
        patches = []
        for box in bboxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                patch = np.random.uniform(0., 255., (128, 64, 3)).astype(np.uint8)
            patches.append(patch)
        if not patches:
            return np.empty((0, 128), dtype=np.float32)
        return self.encode_patches(np.asarray(patches))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--detections", default="det/det.txt")
    args = parser.parse_args()

    extractor = FeatureExtractor(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    for seq_name in sorted(os.listdir(args.mot_dir)):
        seq_dir = os.path.join(args.mot_dir, seq_name)
        det_path = os.path.join(seq_dir, args.detections)
        if not os.path.isfile(det_path):
            print(f"Skipping {seq_name}, no detections.")
            continue

        det_data = np.loadtxt(det_path, delimiter=',', dtype=float)
        if det_data.ndim == 1:
            det_data = det_data.reshape(1, -1)

        detections_by_frame = {}
        for det in det_data:
            frame = int(det[0])
            detections_by_frame.setdefault(frame, []).append(det)

        outputs = []
        for frame, det_list in sorted(detections_by_frame.items()):
            img_path = os.path.join(seq_dir, "img1", f"{frame:06d}.jpg")
            image = cv2.imread(img_path)
            if image is None:
                continue
            bboxes = [(d[2], d[3], d[4], d[5]) for d in det_list]
            features = extractor.extract_features(image, bboxes)
            for det, feat in zip(det_list, features):
                det_info = det[:10]
                outputs.append(np.concatenate([det_info, feat]))
        out_file = os.path.join(args.output_dir, f"{seq_name}.npy")
        np.save(out_file, np.asarray(outputs, dtype=np.float32))
        print(f"Saved: {out_file}")
