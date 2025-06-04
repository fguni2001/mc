import numpy as np
import colorsys
from .image_display import SimpleImageDisplayer

def unique_color(tag, h_step=0.41):
    """
    Generate a unique RGB color (0–255) for a given integer tag using HSV space.
    """
    h = (tag * h_step) % 1
    v = 1.0 - (int(tag * h_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return (int(255 * r), int(255 * g), int(255 * b))


class DummyVisualizer:
    """Fast-forward visualizer: no drawing, just calls your frame callback."""
    def __init__(self, seqdata):
        self.idx = seqdata["min_frame_idx"]
        self.last = seqdata["max_frame_idx"]

    def set_image(self, img): pass
    def draw_groundtruth(self, ids, boxes): pass
    def draw_detections(self, dets): pass
    def draw_trackers(self, trks): pass

    def run(self, fn):
        while self.idx <= self.last:
            fn(self, self.idx)
            self.idx += 1


class Displayer:
    """OpenCV-based visualizer for ground truth, detections, and tracker outputs."""
    def __init__(self, seqdata, ms_delay):
        # Build a 1024px-wide window, preserving aspect ratio.
        # Use actual frame size!
        frame_h, frame_w = seqdata["image_size"][:2]
        display_shape = (frame_w, frame_h)


        self.viewer = SimpleImageDisplayer(ms_delay, display_shape,
                                           f"Seq {seqdata['sequence_name']}")
        self.viewer.line_width = 2

        self.idx = seqdata["min_frame_idx"]
        self.last = seqdata["max_frame_idx"]

    def run(self, callback):
        self.viewer.show(lambda: self._advance(callback))

    def _advance(self, callback):
        if self.idx > self.last:
            return False
        callback(self, self.idx)
        self.idx += 1
        return True

    def set_image(self, img):
        self.viewer.img = img

    def draw_groundtruth(self, ids, boxes):
        """Draw ground-truth boxes in unique colors per ID."""
        self.viewer.line_width = 2
        for id_, box in zip(ids, boxes):
            self.viewer.set_color(unique_color(id_))
            self.viewer.draw_rect(*box.astype(np.int64), lbl=str(id_))

    def draw_detections(self, dets):
        """Draw raw detection boxes in solid red."""
        self.viewer.line_width = 2
        self.viewer.set_color((0, 0, 255))  # BGR = red
        for d in dets:
            self.viewer.draw_rect(*d.tlwh)

    def draw_trackers(self, tracks):
        """Draw tracker boxes with a unique color per track_id."""
        self.viewer.line_width = 2
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            col = unique_color(t.track_id)
            self.viewer.set_color(col)
            self.viewer.draw_rect(*t.to_tlwh().astype(np.int64),
                                  lbl=str(t.track_id))
