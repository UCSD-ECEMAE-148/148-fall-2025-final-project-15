#!/usr/bin/env python3
"""
Person Distance Detection using OAK-D
Uses mono cameras + stereo depth + YOLO person detection

Run with: python -u person_distance.py
For display: python -u person_distance.py --show
"""

import sys
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

SHOW_DISPLAY = "--show" in sys.argv

print("Loading YOLO model...", flush=True)
yolo = YOLO("yolov8n.pt")
print("YOLO loaded!", flush=True)

pipeline = dai.Pipeline()

# Mono cameras for stereo
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)

# Link mono cameras to stereo
monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
monoRightOut = monoRight.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Output queues
leftQueue = stereo.syncedLeft.createOutputQueue()
depthQueue = stereo.depth.createOutputQueue()

if SHOW_DISPLAY:
    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]

print("Starting pipeline...", flush=True)
with pipeline:
    pipeline.start()
    print("Pipeline running! Press Ctrl+C to quit" + (" or 'q' in window" if SHOW_DISPLAY else ""), flush=True)

    frame_count = 0
    try:
        while pipeline.isRunning():
            leftFrame = leftQueue.get()
            depthFrame = depthQueue.get()

            # Get frames
            left = leftFrame.getCvFrame()
            depth = depthFrame.getFrame()

            # Handle grayscale or color frame
            if len(left.shape) == 2:
                frame_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = left

            # Detect people
            results = yolo(frame_bgr, verbose=False, conf=0.4)

            nearest_dist = -1.0
            nearest_cx = -1
            nearest_cy = -1
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 0:  # person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Get depth at center (with bounds check)
                        py = min(max(cy, 0), depth.shape[0] - 1)
                        px = min(max(cx, 0), depth.shape[1] - 1)

                        # Sample small region for robustness
                        y1d, y2d = max(0, py-5), min(depth.shape[0], py+5)
                        x1d, x2d = max(0, px-5), min(depth.shape[1], px+5)
                        region = depth[y1d:y2d, x1d:x2d]
                        valid = region[region > 0]

                        if len(valid) > 0:
                            dist_m = float(np.median(valid)) / 1000.0
                            if nearest_dist < 0 or dist_m < nearest_dist:
                                nearest_dist = dist_m
                                nearest_cx = cx
                                nearest_cy = cy

                            if SHOW_DISPLAY:
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{dist_m:.2f}m"
                                cv2.putText(frame_bgr, label, (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Print nearest person info
            frame_count += 1
            if nearest_dist > 0:
                print(f"[{frame_count}] x={nearest_cx}, y={nearest_cy}, dist={nearest_dist:.2f}m", flush=True)
            else:
                if frame_count % 10 == 0:  # Print every 10 frames if no detection
                    print(f"[{frame_count}] No person detected", flush=True)

            if SHOW_DISPLAY:
                maxDepth = max(1, np.max(depth))
                depthColor = cv2.applyColorMap(((depth / maxDepth) * 255).astype(np.uint8), colorMap)
                cv2.imshow("Person Detection", frame_bgr)
                cv2.imshow("Depth", depthColor)
                if cv2.waitKey(1) == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user")

if SHOW_DISPLAY:
    cv2.destroyAllWindows()
print("Done!")
