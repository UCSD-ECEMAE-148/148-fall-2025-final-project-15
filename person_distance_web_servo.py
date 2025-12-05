#!/usr/bin/env python3
"""
Person Distance Detection with Web Streaming + Laser Servo Control
The laser will track and point at the nearest detected person.

View at http://localhost:5000 (with SSH port forwarding)
Run with: python -u person_distance_web_servo.py
"""

import sys
import cv2
import depthai as dai
import numpy as np
from flask import Flask, Response
import threading
from ultralytics import YOLO

# Servo control
from adafruit_servokit import ServoKit

app = Flask(__name__)

# Global frame storage
output_frame = None
frame_lock = threading.Lock()

# Detection info
detection_info = {"x": -1, "y": -1, "dist": -1.0, "angle_x": 90, "angle_y": 90}

# Camera field of view (OAK-D mono cameras ~71.9° horizontal, ~56.7° vertical)
HFOV = 71.9  # horizontal field of view in degrees
VFOV = 56.7  # vertical field of view in degrees

# Servo channels (from laser.py)
PAN_CHANNEL = 7   # horizontal servo
TILT_CHANNEL = 4  # vertical servo

# Servo smoothing (to avoid jerky movements)
current_pan = 90.0
current_tilt = 90.0
SMOOTHING = 0.3  # 0 = no smoothing, 1 = instant

def setup_servos():
    """Initialize the servo controller"""
    print("Setting up servo controller...", flush=True)
    kit = ServoKit(channels=16)
    
    # Set pulse width range for both servos
    kit.servo[PAN_CHANNEL].set_pulse_width_range(500, 2500)
    kit.servo[TILT_CHANNEL].set_pulse_width_range(500, 2500)
    
    # Center servos on startup
    kit.servo[PAN_CHANNEL].angle = 90
    kit.servo[TILT_CHANNEL].angle = 90
    
    print(f"Servos initialized: pan=ch{PAN_CHANNEL}, tilt=ch{TILT_CHANNEL}", flush=True)
    return kit

def move_servos(kit, target_pan, target_tilt):
    """Move servos smoothly to target angles"""
    global current_pan, current_tilt
    
    # Smooth movement (lerp toward target)
    current_pan = current_pan + SMOOTHING * (target_pan - current_pan)
    current_tilt = current_tilt + SMOOTHING * (target_tilt - current_tilt)
    
    # Clamp to valid range
    pan_angle = max(0, min(180, current_pan))
    tilt_angle = max(0, min(180, current_tilt))
    
    # Move servos
    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle
    
    return pan_angle, tilt_angle

def pixel_to_angle(cx, cy, frame_w, frame_h):
    """
    Convert pixel coordinates to servo angles (0-180 range).
    
    Center of frame = 90° (neutral servo position)
    Left edge = 90 + (HFOV/2) ≈ 126° (servo turns right to look left)
    Right edge = 90 - (HFOV/2) ≈ 54° (servo turns left to look right)
    """
    # Normalize to -0.5 to +0.5 (center = 0)
    nx = (cx / frame_w) - 0.5  # negative = left, positive = right
    ny = (cy / frame_h) - 0.5  # negative = top, positive = bottom
    
    # Convert to angle offset from center
    angle_x = 90 - (nx * HFOV) + 6  # +6 offset for laser mounted on right  # Pan servo angle
    angle_y = 90 - (ny * VFOV)  # Tilt servo angle
    
    # Clamp to valid servo range
    angle_x = max(0, min(180, angle_x))
    angle_y = max(0, min(180, angle_y))
    
    return round(angle_x, 1), round(angle_y, 1)

def run_detection():
    global output_frame, detection_info, current_pan, current_tilt
    
    # Setup servos first
    kit = setup_servos()
    
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
        print("LASER TRACKING ENABLED - will point at nearest person", flush=True)

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

            # Draw frame elements
            # Draw center crosshair
            cv2.line(frame_bgr, (frame_w//2 - 20, frame_h//2), (frame_w//2 + 20, frame_h//2), (100, 100, 100), 1)
            cv2.line(frame_bgr, (frame_w//2, frame_h//2 - 20), (frame_w//2, frame_h//2 + 20), (100, 100, 100), 1)

            if nearest_box:
                x1, y1, x2, y2 = nearest_box
                
                # Calculate servo angles
                angle_x, angle_y = pixel_to_angle(nearest_cx, nearest_cy, frame_w, frame_h)
                
                # MOVE SERVOS to track person!
                actual_pan, actual_tilt = move_servos(kit, angle_x, angle_y)
                
                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point of person
                cv2.circle(frame_bgr, (nearest_cx, nearest_cy), 5, (0, 0, 255), -1)
                
                # Draw line from center to person (laser line visualization)
                cv2.line(frame_bgr, (frame_w//2, frame_h//2), (nearest_cx, nearest_cy), (0, 0, 255), 2)
                
                # Draw info text
                label1 = f"x={nearest_cx} y={nearest_cy} dist={nearest_dist:.2f}m"
                label2 = f"pan={actual_pan:.1f} tilt={actual_tilt:.1f} [TRACKING]"
                cv2.putText(frame_bgr, label1, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame_bgr, label2, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Status text
                cv2.putText(frame_bgr, "LASER TRACKING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No person detected - return to center
                move_servos(kit, 90, 90)
                cv2.putText(frame_bgr, "NO TARGET", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            frame_count += 1
            if nearest_dist > 0:
                angle_x, angle_y = pixel_to_angle(nearest_cx, nearest_cy, frame_w, frame_h)
                detection_info = {
                    "x": nearest_cx, "y": nearest_cy, 
                    "dist": nearest_dist,
                    "angle_x": angle_x, "angle_y": angle_y
                }
                print(f"[{frame_count}] x={nearest_cx}, y={nearest_cy}, dist={nearest_dist:.2f}m | pan={current_pan:.1f}° tilt={current_tilt:.1f}° [TRACKING]", flush=True)
            else:
                detection_info = {"x": -1, "y": -1, "dist": -1.0, "angle_x": 90, "angle_y": 90}
                if frame_count % 10 == 0:
                    print(f"[{frame_count}] No person detected - returning to center", flush=True)

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
    <head><title>Person Distance Detection + Laser Tracking</title></head>
    <body style="background:#111;color:#fff;font-family:monospace;text-align:center">
        <h1>Person Distance Detection + LASER TRACKING</h1>
        <img src="/video_feed" style="max-width:100%">
        <p style="color:#0ff">LASER IS ACTIVE - tracking nearest person!</p>
        <p>Green box = nearest person | Red dot/line = laser target</p>
        <p>pan = horizontal servo | tilt = vertical servo</p>
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
    print("LASER TRACKING MODE", flush=True)
    print("Open in browser: http://localhost:5000", flush=True)
    print("(with SSH port forwarding)", flush=True)
    print("="*50 + "\n", flush=True)
    
    app.run(host="0.0.0.0", port=5000, threaded=True)
