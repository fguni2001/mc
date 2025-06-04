import numpy as np
import cv2

def nms_boxes(bboxes, overlap_thresh, conf_scores=None):
    if len(bboxes) == 0:
        return []
    bboxes = bboxes.astype(np.float64)
    x1, y1 = bboxes[:,0], bboxes[:,1]
    x2, y2 = bboxes[:,0]+bboxes[:,2], bboxes[:,1]+bboxes[:,3]
    area = (x2-x1+1)*(y2-y1+1)
    if conf_scores is not None:
        order = np.argsort(conf_scores)
    else:
        order = np.argsort(y2)
    selected = []
    while len(order) > 0:
        i = order[-1]
        selected.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h) / area[order[:-1]]
        keep = np.where(overlap <= overlap_thresh)[0]
        order = order[np.append(keep, -1)]
        order = order[:-1]
    return selected
