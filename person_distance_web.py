#!/usr/bin/env python3
"""
Person Distance Detection with Web Streaming
View at http://localhost:5000 (with SSH port forwarding)

Run with: python -u person_distance_web.py
"""

import sys
import cv2
import depthai as dai
import numpy as np
from flask import Flask, Response
import threading
from ultralytics import YOLO

app = Flask(__name__)

# Global frame storage
output_frame = None
frame_lock = threading.Lock()

# Detection info
detection_info = {"x": -1, "y": -1, "dist": -1.0, "angle_x": 90, "angle_y": 90}

# Camera field of view (OAK-D mono cameras ~71.9° horizontal, ~56.7° vertical)
HFOV = 71.9  # horizontal field of view in degrees
VFOV = 56.7  # vertical field of view in degrees

def pixel_to_angle(cx, cy, frame_w, frame_h):
    """
    Convert pixel coordinates to servo angles (0-180 range).
    
    Center of frame = 90° (neutral servo position)
    Left edge = 90 + (HFOV/2) ≈ 126° (servo turns right to look left)
    Right edge = 90 - (HFOV/2) ≈ 54° (servo turns left to look right)
    
    For pan (horizontal):
      - Person on LEFT of frame → positive angle offset → servo > 90°
      - Person on RIGHT of frame → negative angle offset → servo < 90°
    
    For tilt (vertical):
      - Person ABOVE center → positive angle offset → servo > 90°
      - Person BELOW center → negative angle offset → servo < 90°
    """
    # Normalize to -0.5 to +0.5 (center = 0)
    nx = (cx / frame_w) - 0.5  # negative = left, positive = right
    ny = (cy / frame_h) - 0.5  # negative = top, positive = bottom
    
    # Convert to angle offset from center
    # Person on right (nx > 0) → servo angle decreases (looks right)
    # Person on left (nx < 0) → servo angle increases (looks left)
    angle_x = 90 - (nx * HFOV)  # Pan servo angle
    
    # Person below (ny > 0) → servo angle decreases (looks down)
    # Person above (ny < 0) → servo angle increases (looks up)
    angle_y = 90 - (ny * VFOV)  # Tilt servo angle
    
    # Clamp to valid servo range
    angle_x = max(0, min(180, angle_x))
    angle_y = max(0, min(180, angle_y))
    
    return round(angle_x, 1), round(angle_y, 1)

def run_detection():
    global output_frame, detection_info
    
    print("Loading YOLO model...", flush=True)
    yolo = YOLO("yolov8n.pt")
    print("YOLO loaded!", flush=True)

    pipeline = dai.Pipeline()

    # Mono cameras for stereo
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)

    monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
    monoRightOut = monoRight.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    leftQueue = stereo.syncedLeft.createOutputQueue()
    depthQueue = stereo.depth.createOutputQueue()

    print("Starting pipeline...", flush=True)
    with pipeline:
        pipeline.start()
        print("Pipeline running!", flush=True)
        print(f"FOV: {HFOV}° horizontal, {VFOV}° vertical", flush=True)
        print("Angles: 90° = center, >90° = left/up, <90° = right/down", flush=True)

        frame_count = 0
        while pipeline.isRunning():
            leftFrame = leftQueue.get()
            depthFrame = depthQueue.get()

            left = leftFrame.getCvFrame()
            depth = depthFrame.getFrame()

            if len(left.shape) == 2:
                frame_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = left

            frame_h, frame_w = frame_bgr.shape[:2]
            
            results = yolo(frame_bgr, verbose=False, conf=0.4)

            nearest_dist = -1.0
            nearest_cx = -1
            nearest_cy = -1
            nearest_box = None
            
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        py = min(max(cy, 0), depth.shape[0] - 1)
                        px = min(max(cx, 0), depth.shape[1] - 1)

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
                                nearest_box = (x1, y1, x2, y2)

            # Draw only the nearest person
            if nearest_box:
                x1, y1, x2, y2 = nearest_box
                
                # Calculate servo angles
                angle_x, angle_y = pixel_to_angle(nearest_cx, nearest_cy, frame_w, frame_h)
                
                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center crosshair
                cv2.line(frame_bgr, (frame_w//2 - 20, frame_h//2), (frame_w//2 + 20, frame_h//2), (100, 100, 100), 1)
                cv2.line(frame_bgr, (frame_w//2, frame_h//2 - 20), (frame_w//2, frame_h//2 + 20), (100, 100, 100), 1)
                
                # Draw center point of person
                cv2.circle(frame_bgr, (nearest_cx, nearest_cy), 5, (0, 0, 255), -1)
                
                # Draw info text
                label1 = f"x={nearest_cx} y={nearest_cy} dist={nearest_dist:.2f}m"
                label2 = f"pan={angle_x:.1f} tilt={angle_y:.1f}"
                cv2.putText(frame_bgr, label1, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_bgr, label2, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            frame_count += 1
            if nearest_dist > 0:
                angle_x, angle_y = pixel_to_angle(nearest_cx, nearest_cy, frame_w, frame_h)
                detection_info = {
                    "x": nearest_cx, "y": nearest_cy, 
                    "dist": nearest_dist,
                    "angle_x": angle_x, "angle_y": angle_y
                }
                print(f"[{frame_count}] x={nearest_cx}, y={nearest_cy}, dist={nearest_dist:.2f}m | pan={angle_x:.1f}° tilt={angle_y:.1f}°", flush=True)
            else:
                detection_info = {"x": -1, "y": -1, "dist": -1.0, "angle_x": 90, "angle_y": 90}
                if frame_count % 10 == 0:
                    print(f"[{frame_count}] No person detected", flush=True)

            with frame_lock:
                output_frame = frame_bgr.copy()

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return """
    <html>
    <head><title>Person Distance Detection</title></head>
    <body style="background:#111;color:#fff;font-family:monospace;text-align:center">
        <h1>Person Distance Detection</h1>
        <img src="/video_feed" style="max-width:100%">
        <p>Green box = nearest person | Red dot = center</p>
        <p>pan = horizontal servo angle | tilt = vertical servo angle</p>
        <p>90° = center | >90° = left/up | <90° = right/down</p>
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Start detection in background thread
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    
    print("\n" + "="*50, flush=True)
    print("Open in browser: http://localhost:5000", flush=True)
    print("(with SSH port forwarding)", flush=True)
    print("="*50 + "\n", flush=True)
    
    app.run(host="0.0.0.0", port=5000, threaded=True)
