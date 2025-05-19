import os
from dotenv import load_dotenv
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if platform.system() == 'Darwin': 
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

load_dotenv()

app = Flask(__name__)


sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
email_password = os.getenv("EMAIL_PASSWORD")


if not all([sender_email, receiver_email, email_password]):
    logger.error("Email configuration is incomplete. Please check your .env file")
    logger.error(f"Sender Email: {'Set' if sender_email else 'Not set'}")
    logger.error(f"Receiver Email: {'Set' if receiver_email else 'Not set'}")
    logger.error(f"Email Password: {'Set' if email_password else 'Not set'}")

model = YOLO("Model/ppe.pt")


MAX_VIOLATIONS = 50
stats = {
    'hardhat_count': 0,
    'vest_count': 0,
    'person_count': 0,
    'mask_count': 0,
    'violations': deque(maxlen=MAX_VIOLATIONS),
    'last_update': datetime.now().isoformat(),
    'camera_status': 'initializing',
    'email_status': 'ready',
    'total_violations': 0
}


last_email_time = time.time()
email_sent_flag = False
email_sent_time = 0


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

def send_email_alert(image_path):
    """Send email alert with attached image"""
    try:

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "Alert: Safety Violation Detected!"
        

        body = """
        A safety violation was detected in the construction site.
        
        Details:
        - Time: {time}
        - Violation Type: No Hardhat
        - Number of People: {person_count}
        
        Please find the attached image showing the situation.
        """.format(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            person_count=stats['person_count']
        )
        message.attach(MIMEText(body, "plain"))
        
  
        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename=violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            message.attach(part)
        
     
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        
        logger.info("Email alert sent successfully")
        stats['email_status'] = 'sent'
        return True
        
    except smtplib.SMTPAuthenticationError:
        logger.error("Email authentication failed. Please check your email credentials")
        stats['email_status'] = 'auth_error'
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error occurred: {str(e)}")
        stats['email_status'] = 'smtp_error'
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        stats['email_status'] = 'error'
    
    return False

def send_email_in_background(image_path):
    """Send email in a background thread"""
    email_thread = threading.Thread(target=send_email_alert, args=(image_path,))
    email_thread.daemon = True 
    email_thread.start()

def process_frame(frame):
    global stats, last_email_time, email_sent_flag, email_sent_time
    
  
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

    if person_detected and violations_detected and (time.time() - last_email_time) >= 100:
        try:

            image_path = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(image_path, frame)
            
        
            for violation_type in violations_detected:
                send_email_in_background(image_path)
                email_sent_flag = True
                email_sent_time = time.time()
                last_email_time = time.time()
                
              
                violation = {
                    'id': stats['total_violations'] + 1,
                    'timestamp': datetime.now().isoformat(),
                    'type': violation_type,
                    'person_count': person_count,
                    'email_status': stats['email_status'],
                    'image_path': image_path
                }
                
               
                stats['violations'].append(violation)
                stats['total_violations'] += 1
                
                logger.info(f"Safety violation detected: {violation_type} and email sent. Image saved as {image_path}")
        except Exception as e:
            logger.error(f"Error processing violation: {str(e)}")
            stats['email_status'] = 'error'

    
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

    if email_sent_flag and (time.time() - email_sent_time) < 3:
        draw_text_with_background(frame, "Email Sent", (frame.shape[1] - 100, 30), font_scale=0.5, color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.8, padding=5)

    return frame

def get_video_source():
    global video_capture, current_video_source, is_uploaded_video
    if current_video_source is None:
        video_capture = cv2.VideoCapture(0)
        is_uploaded_video = False
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
    

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_file.name)


    release_video_source()
    

    current_video_source = temp_file.name
    video_capture = cv2.VideoCapture(current_video_source)
    
    if not video_capture.isOpened():
        return jsonify({'error': 'Could not open video file'}), 400


    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    is_uploaded_video = True
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    })

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