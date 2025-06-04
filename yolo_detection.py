# yolo_detection.py

import torch
import cv2
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

class YOLOv3Detector:
    def __init__(
        self,
        weights_path="yolov3.pt",
        device="cpu",
        conf_thres=0.25,
        iou_thres=0.45,
        img_size=640
    ):
        self.device = select_device(device)
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"YOLOv3 weights not found at {weights_path}")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.model.eval()
        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = check_img_size((img_size, img_size), s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.target_class_id = 0  

    def preprocess(self, frame):
        img = cv2.resize(frame, self.imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, frame):
        im = self.preprocess(frame)
        with torch.no_grad():
            pred = self.model(im)[0]
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres,
                classes=[self.target_class_id]
            )[0]

        if pred is None or len(pred) == 0:
            return []

        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], frame.shape).round()
        results = []
        for *xyxy, conf, cls in pred.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            results.append([x1, y1, w, h, float(conf)])

        return results
