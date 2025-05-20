import os
from dotenv import load_dotenv
import cv2
import time
from ultralytics import YOLO
import threading
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
import base64
from io import BytesIO
from PIL import Image
import json
from collections import defaultdict, deque
from datetime import datetime
import platform
import logging
import tempfile

from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=30)

already_reported_ids = {}


from azure.storage.blob import BlobServiceClient, ContentSettings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if platform.system() == 'Darwin': 
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'violation-images')

blob_service_client = None
if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

        if not container_client.exists():
            container_client.create_container()
        logger.info("Azure Blob Storage connection established successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure Blob Storage: {str(e)}")
        blob_service_client = None

app = Flask(__name__)

try:
    import torch
    model = YOLO("Model/ppe.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    model = None

MAX_VIOLATIONS = 50
stats = {
    'hardhat_count': 0,
    'vest_count': 0,
    'person_count': 0,
    'mask_count': 0,
    'violations': deque(maxlen=MAX_VIOLATIONS),
    'last_update': datetime.now().isoformat(),
    'camera_status': 'initializing',
    'total_violations': 0
}

current_video_source = None
video_capture = None
is_uploaded_video = False

def draw_text_with_background(frame, text, position, font_scale=0.4, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.7, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    overlay = frame.copy()
    x, y = position
    cv2.rectangle(overlay, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def process_frame(frame):
    global stats, already_reported_ids

    if model is None:
        draw_text_with_background(frame, "Model not loaded", (200, 240), font_scale=1, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.8, padding=10)
        return frame

    hardhat_count = 0
    vest_count = 0
    person_count = 0
    mask_count = 0
    violations_detected = []

    results = model(frame)
    detections = []

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label == 'Person':
                    detections.append(((x1, y1, x2, y2), conf, label, None))

    tracks = tracker.update_tracks(detections, frame=frame)
    frame_copy = frame.copy()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                color = (0, 255, 255)

                if label == "Hardhat":
                    hardhat_count += 1
                elif label == "NO-Hardhat":
                    violations_detected.append(("No Hardhat", (x1, y1, x2, y2)))
                elif label == "Safety Vest":
                    vest_count += 1
                elif label == "NO-Safety Vest":
                    violations_detected.append(("No Safety Vest", (x1, y1, x2, y2)))
                elif label == "Mask":
                    mask_count += 1
                elif label == "NO-Mask":
                    violations_detected.append(("No Mask", (x1, y1, x2, y2)))
                elif label == "Person":
                    person_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text_with_background(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), font_scale=0.4, color=(255, 255, 255), bg_color=color, alpha=0.8, padding=4)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        if track_id not in already_reported_ids:
            already_reported_ids[track_id] = set()

        for (violation_type, (vx1, vy1, vx2, vy2)) in violations_detected:
            if vx1 >= l and vx2 <= r and vy1 >= t and vy2 <= b:
                if violation_type not in already_reported_ids[track_id]:
                    already_reported_ids[track_id].add(violation_type)
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        blob_name = f"violation_{track_id}_{timestamp}.jpg"
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                            cv2.imwrite(temp_file.name, frame_copy)
                            image_url = None
                            if blob_service_client:
                                container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
                                with open(temp_file.name, "rb") as data:
                                    blob_client = container_client.get_blob_client(blob_name)
                                    blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="image/jpeg"))
                                    image_url = blob_client.url
                            os.unlink(temp_file.name)

                        violation = {
                            'id': stats['total_violations'] + 1,
                            'timestamp': datetime.now().isoformat(),
                            'type': violation_type,
                            'person_count': person_count,
                            'image_url': image_url
                        }

                        stats['violations'].append(violation)
                        stats['total_violations'] += 1
                        logger.info(f"Violation recorded: {violation_type} by track_id {track_id}")
                    except Exception as e:
                        logger.error(f"Error processing violation: {str(e)}")
                break

    stats['hardhat_count'] = hardhat_count
    stats['vest_count'] = vest_count
    stats['mask_count'] = mask_count
    stats['person_count'] = person_count
    stats['last_update'] = datetime.now().isoformat()
    stats['camera_status'] = 'active'

    y_position = 30
    for text in [
        f"Hardhats: {hardhat_count}",
        f"Safety Vests: {vest_count}",
        f"Masks: {mask_count}",
        f"People: {person_count}",
        f"Total Violations: {stats['total_violations']}"
    ]:
        draw_text_with_background(frame, text, (10, y_position), font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7, padding=5)
        y_position += 30

    return frame


def get_video_source():
    global video_capture, current_video_source, is_uploaded_video
    if current_video_source is None:
        try:
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                logger.warning("No camera available. Please upload a video file instead.")
                # Create an informative frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Available", (100, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Please upload a video file", (80, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "or use a local deployment", (100, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                video_capture = None
                current_video_source = "blank"
            is_uploaded_video = False
        except Exception as e:
            logger.error(f"Error accessing camera: {str(e)}")
            video_capture = None
            current_video_source = "blank"
    return video_capture

def release_video_source():
    global video_capture, current_video_source
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    current_video_source = None

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_source, video_capture, is_uploaded_video
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Create a temporary file with a unique name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_file.name)
        temp_file.close()  # Close the file to ensure it's written

        # Release any existing video source
        release_video_source()
        
        # Set up the new video source
        current_video_source = temp_file.name
        video_capture = cv2.VideoCapture(current_video_source)
        
        if not video_capture.isOpened():
            logger.error(f"Could not open video file: {current_video_source}")
            return jsonify({'error': 'Could not open video file'}), 400

        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Test read first frame
        ret, frame = video_capture.read()
        if not ret:
            logger.error("Failed to read first frame from uploaded video")
            return jsonify({'error': 'Invalid video file'}), 400
            
        # Reset to beginning
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        is_uploaded_video = True
        logger.info(f"Video uploaded successfully: {current_video_source}")
        logger.info(f"Video properties: FPS={fps}, Frames={frame_count}, Duration={duration}s")
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        })
    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/switch_to_camera', methods=['POST'])
def switch_to_camera():
    global current_video_source, video_capture, is_uploaded_video
    
    release_video_source()
    
    current_video_source = None
    video_capture = cv2.VideoCapture(0)
    is_uploaded_video = False
    
    return jsonify({'message': 'Switched to camera feed'})

def generate_frames():
    global video_capture, is_uploaded_video
    
    logger.info("Starting frame generation")
    cap = get_video_source()
    
    if cap is None:
        logger.error("No video source available")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Video Source", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {current_video_source}")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Failed to open video", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video source opened successfully. FPS: {fps}, Frames: {frame_count}, Duration: {duration}s")
        
        target_fps = 15  # Limit to 15 FPS for better performance
        frame_interval = int(fps / target_fps) if fps > target_fps else 1
        frame_counter = 0
        last_frame_time = time.time()

        while True:
            try:
                # Calculate time to wait for next frame
                current_time = time.time()
                time_to_wait = (1.0 / target_fps) - (current_time - last_frame_time)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

                ret, frame = cap.read()
                if not ret:
                    if is_uploaded_video:
                        logger.info("End of video reached, restarting...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.error("Failed to read frame")
                    break

                frame_counter += 1
                if frame_counter % frame_interval == 0:
                    try:
                        # Process frame
                        processed_frame = process_frame(frame)
                        
                        # Encode frame
                        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not ret:
                            logger.error("Failed to encode frame")
                            continue
                            
                        frame = buffer.tobytes()
                        last_frame_time = time.time()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error processing frame: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error in frame generation loop: {str(e)}")
                break

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
    finally:
        logger.info("Releasing video capture")
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video feed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    stats_copy = stats.copy()
    stats_copy['violations'] = list(stats_copy['violations'])
    return jsonify(stats_copy)

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'port': os.environ.get('PORT', 10000),
        'host': '0.0.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on host 0.0.0.0 and port {port}")
    logger.info(f"Environment variables: PORT={os.environ.get('PORT')}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 