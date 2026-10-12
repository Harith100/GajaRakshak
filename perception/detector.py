"""YOLOX object detector on ONNX Runtime — the real model that replaces sim/sensors.py on a tower.

Model: YOLOX-S (Megvii, Apache-2.0), COCO-pretrained. COCO includes "elephant", so it detects
elephants in RGB images out of the box. Thermal imagery needs fine-tuning on phase-0 data —
the pipeline around the model does not change.

On the tower this runs as a TensorRT INT8 engine on a Jetson Orin; here it runs on CPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]

# COCO class → GajaRakshak label. Only "elephant" counts as elephant evidence at the edge.
LABEL_MAP = {"elephant": "elephant", "cow": "cattle", "person": "human", "horse": "other_animal",
             "sheep": "other_animal", "dog": "other_animal", "bear": "other_animal", "zebra": "other_animal",
             "giraffe": "other_animal", "train": "vehicle", "car": "vehicle", "truck": "vehicle",
             "motorcycle": "vehicle", "bus": "vehicle"}


class YoloxDetector:
    def __init__(self, model_path: str | Path, input_size: int = 640, conf: float = 0.25, nms: float = 0.45):
        avail = ort.get_available_providers()
        providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in avail] + \
                    ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = input_size
        self.conf, self.nms = conf, nms
        self.provider = self.sess.get_providers()[0]
        self._grids, self._strides = self._make_grids(input_size)

    @staticmethod
    def _make_grids(size: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(g)
            strides.append(np.full((1, g.shape[1], 1), s))
        return np.concatenate(grids, 1).astype(np.float32), np.concatenate(strides, 1).astype(np.float32)

    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None], r

    def __call__(self, img_bgr: np.ndarray) -> tuple[list[dict], float]:
        """Returns detections [{coco, label, conf, box:(x1,y1,x2,y2)}] in image pixels, and latency in ms."""
        t0 = time.perf_counter()
        x, r = self._preprocess(img_bgr)
        out = self.sess.run(None, {self.input_name: x})[0][0]            # (8400, 85)
        out[:, :2] = (out[:, :2] + self._grids[0]) * self._strides[0]
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides[0]
        cls = out[:, 5:].argmax(1)
        score = out[:, 4] * out[np.arange(len(out)), 5 + cls]
        keep = score >= self.conf
        b, s, c = out[keep, :4], score[keep], cls[keep]
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1) / r   # x, y, w, h
        dets = []
        if len(boxes):
            # class-aware NMS: offset boxes per class so classes never suppress each other
            off = (c[:, None] * 4096).astype(np.float32)
            nms_boxes = np.concatenate([boxes[:, :2] + off, boxes[:, 2:]], 1).tolist()
            idx = cv2.dnn.NMSBoxes(nms_boxes, s.tolist(), self.conf, self.nms)
            h, w = img_bgr.shape[:2]
            for i in np.array(idx).reshape(-1):
                x0, y0, bw, bh = boxes[i]
                name = COCO[int(c[i])]
                dets.append({"coco": name, "label": LABEL_MAP.get(name, "other"), "conf": float(s[i]),
                             "box": (max(0.0, x0), max(0.0, y0), min(w - 1.0, x0 + bw), min(h - 1.0, y0 + bh))})
        dets.sort(key=lambda d: -d["conf"])
        return dets, (time.perf_counter() - t0) * 1000
