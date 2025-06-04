# viewer_module.py
"""
Viewer utilities based on OpenCV for displaying and drawing on images.
"""
import cv2
import numpy as np
import time

def roi_in_bounds(image, roi):
    # Determines if a region-of-interest is fully inside an image.
    x, y, w, h = roi
    H, W = image.shape[:2]
    return (0 <= x < W) and (0 <= y < H) and (x + w <= W) and (y + h <= H)

def get_roi(image, roi):
    # Returns the selected region from image, assumes ROI is valid.
    x, y, w, h = roi
    return image[y:y+h, x:x+w] if image.ndim == 2 else image[y:y+h, x:x+w, :]

class SimpleImageDisplayer:
    """
    Simple display and draw class for images (OpenCV).
    Hotkeys: [SPACE]=pause/resume, [ESC]=exit.
    """
    def __init__(self, interval_ms, win_size=(640,480), title="Window"):
        self.size = win_size
        self.caption = title
        self.delay = interval_ms
        self.writer = None
        self.quit_flag = False

        self.img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.pen_color = (0,0,0)
        self.txt_color = (255,255,255)
        self.line_width = 1

    def set_color(self, color_tuple):
        assert len(color_tuple) == 3
        self.pen_color = tuple(map(int, color_tuple))

    def draw_rect(self, x, y, w, h, lbl=None):
        cv2.rectangle(self.img, (int(x),int(y)), (int(x+w),int(y+h)), self.pen_color, self.line_width)
        if lbl:
            font_scale = 1
            size, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.line_width)
            pt = (int(x)+5, int(y)+5+size[1])
            cv2.rectangle(self.img, (int(x),int(y)), (int(x)+10+size[0], int(y)+10+size[1]), self.pen_color, -1)
            cv2.putText(self.img, lbl, pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), self.line_width)

    def draw_circle(self, x, y, r, lbl=None):
        img_dim = int(r+self.line_width+2)
        roi = (int(x-img_dim), int(y-img_dim), int(2*img_dim), int(2*img_dim))
        if not roi_in_bounds(self.img, roi):
            return
        region = get_roi(self.img, roi)
        center = region.shape[1]//2, region.shape[0]//2
        cv2.circle(region, center, int(r+.5), self.pen_color, self.line_width)
        if lbl:
            cv2.putText(self.img, lbl, center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.txt_color, 2)

    def draw_ellipse(self, mean, cov, lbl=None):
        vals, vecs = np.linalg.eigh(5.99 * cov)
        idx = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[idx]), vecs[:,idx]
        ctr = (int(mean[0]+.5), int(mean[1]+.5))
        axes = (int(vals[0]+.5), int(vals[1]+.5))
        angle = int(180 * np.arctan2(vecs[1,0], vecs[0,0]) / np.pi)
        cv2.ellipse(self.img, ctr, axes, angle, 0, 360, self.pen_color, 2)
        if lbl:
            cv2.putText(self.img, lbl, ctr, cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.txt_color, 2)

    def put_text(self, x, y, txt):
        cv2.putText(self.img, txt, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.txt_color, 2)

    def enable_writer(self, filename, codec="XVID", fps=None):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        fps = fps or int(1000. / self.delay)
        height, width = self.size[1], self.size[0]
        # Defensive: don't allow absurd sizes
        if width > 5000 or height > 5000 or width <= 0 or height <= 0:
            raise ValueError(f"Refusing to write video with abnormal size: {width}x{height}")
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))



    def disable_writer(self):
        self.writer = None

    def show(self, step_fn=None):
        # Main display loop. Handles pause, resume, step, and exit.
        if step_fn:
            update = step_fn
        else:
            update = lambda: True
        paused = False
        while not self.quit_flag:
            t0 = time.time()
            if not paused:
                self.quit_flag = not update()
                if self.writer:
                    self.writer.write(cv2.resize(self.img, (self.size[0], self.size[1])))
            wait = max(1, int(self.delay-1000*(time.time()-t0)))
            cv2.imshow(self.caption, cv2.resize(self.img, (self.size[0], self.size[1])))
            key = cv2.waitKey(wait) & 0xFF
            if key == 27:
                print("Exiting viewer.")
                self.quit_flag = True
            elif key == 32:
                paused = not paused
            elif key == ord('s'):
                self.quit_flag = not update()
                paused = True
        cv2.destroyWindow(self.caption)
        cv2.waitKey(1)
        self.img[:] = 0
        cv2.imshow(self.caption, self.img)

    def stop(self):
        self.quit_flag = True
