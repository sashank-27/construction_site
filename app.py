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
import torch
from types import SimpleNamespace
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

from azure.storage.blob import BlobServiceClient, ContentSettings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if platform.system() == 'Darwin': 
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

load_dotenv()

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'violation-images')

# Initialize Azure Blob Service Client
blob_service_client = None
if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
        logger.info("Azure Blob Storage connection established successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure Blob Storage: {str(e)}")
        blob_service_client = None

app = Flask(__name__)

# Add IterableSimpleNamespace for compatibility
class IterableSimpleNamespace(SimpleNamespace):
    def __iter__(self):
        return iter(vars(self).items())

# Initialize model with error handling
try:
    logger.info("Loading YOLO model...")
    model_path = os.path.join("Model", "ppe.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Add required classes to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([DetectionModel, Sequential])
    
    # Load model with specific version compatibility
    model = YOLO(model_path)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    raise

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
    global stats
    
    hardhat_count = 0
    vest_count = 0
    person_count = 0
    mask_count = 0
    no_hardhat_detected = False
    no_vest_detected = False
    no_mask_detected = False
    person_detected = False
    violations_detected = []

    # Colors 
    colors = [
        (255, 0, 0),  
        (0, 255, 0),  
        (0, 0, 255),  
        (255, 255, 0),  
        (255, 0, 255),  
        (0, 255, 255),  
        (128, 0, 128),  
        (128, 128, 0), 
        (0, 128, 128), 
        (128, 128, 128)  
    ]

    results = model(frame)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} ({confidence:.2f})"

                color = colors[cls % len(colors)]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.4, color=(255, 255, 255), bg_color=color, alpha=0.8, padding=4)

                # Track detections
                if model.names[cls] == "Hardhat":
                    hardhat_count += 1
                elif model.names[cls] == "NO-Hardhat":
                    no_hardhat_detected = True
                    violations_detected.append("No Hardhat")
                elif model.names[cls] == "Safety Vest":
                    vest_count += 1
                elif model.names[cls] == "NO-Safety Vest":
                    no_vest_detected = True
                    violations_detected.append("No Safety Vest")
                elif model.names[cls] == "Mask":
                    mask_count += 1
                elif model.names[cls] == "NO-Mask":
                    no_mask_detected = True
                    violations_detected.append("No Mask")
                elif model.names[cls] == "Person":
                    person_count += 1
                    person_detected = True

    stats['hardhat_count'] = hardhat_count
    stats['vest_count'] = vest_count
    stats['person_count'] = person_count
    stats['mask_count'] = mask_count
    stats['last_update'] = datetime.now().isoformat()
    stats['camera_status'] = 'active'

    if person_detected and violations_detected:
        try:
            # Generate a unique blob name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob_name = f"violation_{timestamp}.jpg"
            
            # Save image to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, frame)
                
                # Upload to Azure Blob Storage if configured
                if blob_service_client:
                    try:
                        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
                        with open(temp_file.name, "rb") as data:
                            blob_client = container_client.get_blob_client(blob_name)
                            blob_client.upload_blob(
                                data,
                                overwrite=True,
                                content_settings=ContentSettings(content_type="image/jpeg")
                            )
                        image_url = blob_client.url
                        logger.info(f"Image uploaded to Azure Blob Storage: {image_url}")
                    except Exception as e:
                        logger.error(f"Failed to upload image to Azure Blob Storage: {str(e)}")
                        image_url = None
                else:
                    image_url = None
                
                # Clean up temporary file
                os.unlink(temp_file.name)
            
            for violation_type in violations_detected:
                violation = {
                    'id': stats['total_violations'] + 1,
                    'timestamp': datetime.now().isoformat(),
                    'type': violation_type,
                    'person_count': person_count,
                    'image_url': image_url
                }
                
                stats['violations'].append(violation)
                stats['total_violations'] += 1
                
                logger.info(f"Safety violation detected: {violation_type}")
        except Exception as e:
            logger.error(f"Error processing violation: {str(e)}")

    sideboard_text = [
        f"Hardhats: {hardhat_count}",
        f"Safety Vests: {vest_count}",
        f"Masks: {mask_count}",
        f"People: {person_count}",
        f"Total Violations: {stats['total_violations']}"
    ]

    y_position = 30
    for text in sideboard_text:
        draw_text_with_background(frame, text, (10, y_position), font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7, padding=5)
        y_position += 30

    return frame

def get_video_source():
    global video_capture, current_video_source, is_uploaded_video
    if current_video_source is None:
        video_capture = cv2.VideoCapture(0)
        is_uploaded_video = False
    return video_capture

def release_video_source():
    global video_capture, current_video_source
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        if current_video_source is not None and os.path.exists(current_video_source):
            os.unlink(current_video_source)
        current_video_source = None
        stats['camera_status'] = 'inactive'
    except Exception as e:
        logger.error(f"Error releasing video source: {str(e)}")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_source, video_capture, is_uploaded_video
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Create a temporary file to store the uploaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_file.name)
        
        # Release any existing video source
        release_video_source()
        
        # Set up the new video source
        current_video_source = temp_file.name
        video_capture = cv2.VideoCapture(current_video_source)
        
        if not video_capture.isOpened():
            os.unlink(temp_file.name)  # Clean up the temp file
            return jsonify({'error': 'Could not open video file'}), 400
        
        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        is_uploaded_video = True
        stats['camera_status'] = 'active'
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        if 'temp_file' in locals():
            os.unlink(temp_file.name)  # Clean up the temp file
        return jsonify({'error': 'Failed to process video file'}), 500

@app.route('/switch_to_camera', methods=['POST'])
def switch_to_camera():
    global current_video_source, video_capture, is_uploaded_video
    
    try:
        release_video_source()
        
        current_video_source = None
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            return jsonify({'error': 'Could not access camera'}), 400
            
        is_uploaded_video = False
        stats['camera_status'] = 'active'
        
        return jsonify({
            'message': 'Switched to camera feed',
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error switching to camera: {str(e)}")
        return jsonify({'error': 'Failed to switch to camera'}), 500

def generate_frames():
    global video_capture, is_uploaded_video
    
    cap = get_video_source()
    if not cap.isOpened():
        print("Error: Unable to access the video source.")
        stats['camera_status'] = 'error'
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_text_with_background(frame, "Video Source Error", (200, 240), font_scale=1, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.8, padding=10)
        draw_text_with_background(frame, "Please check video source", (150, 280), font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.8, padding=10)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    target_fps = 15  
    frame_interval = int(fps / target_fps) if fps > target_fps else 1
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_uploaded_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_counter += 1
        if frame_counter % frame_interval == 0:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(1.0 / target_fps)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    stats_copy = stats.copy()
    stats_copy['violations'] = list(stats_copy['violations'])
    return jsonify(stats_copy)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 