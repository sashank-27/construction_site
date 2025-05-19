![Header Image](Visuals/ppe-public-view.png)  

# Construction Safety Detection - Mail Alert (Yolov8)

This project focuses on enhancing construction site safety through real-time detection of safety gear such as helmets, vests, and masks worn by workers, as well as detecting the presence of a person. The detection is performed using YOLOv8, a state-of-the-art object detection algorithm.


## Overview

Construction sites present various safety hazards, and ensuring that workers wear appropriate safety gear is crucial for accident prevention. This project automates the process of safety gear detection using computer vision techniques. By deploying YOLOv8, the system can detect whether a worker is wearing a helmet, a vest, a mask, or all, and identify people in real-time.

## Features

- **Helmet Detection:** Detects whether a worker is wearing a helmet.
- **Vest Detection:** Detects whether a worker is wearing a safety vest.
- **Mask Detection:** Detects whether a worker is wearing a mask.
- **Person Detection:** Detects the presence of a person within the construction site.
- **Count Display:** Displays real-time counts of detected helmets, vests, masks, and persons on a sideboard overlay.
- **Email Alerts:** Sends email alerts if a person is detected without a helmet, with a frame of the incident attached.
- **Non-Blocking Email Process:** Ensures video feed remains smooth while email alerts are sent in the background.
- **Mail Sent Notification:** A popup is displayed in the top-right corner of the video feed when an email alert is successfully sent.

## Requirements

- Python 3.9
- YOLOv8 dependencies (refer to YOLOv8 documentation for installation instructions)
- OpenCV
- Other dependencies as mentioned in the project code

## Installation

### Using `conda` (Recommended)

1. Clone the repository:

    ```bash
    git clone https://github.com/Ansarimajid/Construction-PPE-Detection.git
    cd Construction-PPE-Detection
    ```

2. Create a conda environment from the `yolo_env.yml` file:

    ```bash
    conda env create -f yolo_env.yml
    ```

3. Activate the environment:

    ```bash
    conda activate yolo
    ```

4. Ensure the YOLOv8 weights file (`ppe.pt`) and place it in the designated directory.

### Using `pip`

1. Clone the repository:

    ```bash
    git clone https://github.com/Ansarimajid/Construction-PPE-Detection.git
    cd Construction-PPE-Detection
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the YOLOv8 weights file (`ppe.pt`) and place it in the designated directory.

---

## Configuration for Email Alerts

To enable email alert functionality, update the `.env` file in the project directory with your email details:

```text
SENDER_EMAIL=your_email@gmail.com
RECEIVER_EMAIL=receiver_email@example.com
EMAIL_PASSWORD=your_email_password
```

- **SENDER_EMAIL:** The email address that will send the alerts.
- **RECEIVER_EMAIL:** The email address that will receive the alerts.
- **EMAIL_PASSWORD:** The app-specific password or account password for the sender email. (For Gmail, you need to generate an [app-specific password](https://support.google.com/accounts/answer/185833?hl=en)).

> **Important:** Do not share your `.env` file publicly to avoid exposing sensitive information.

---

## Usage

1. Navigate to the project directory.

2. Run the detection script:

    ```bash
    python webcam.py
    ```

3. The script will initiate real-time detection using your webcam or process a video file.

4. Detected objects will be highlighted with bounding boxes indicating whether a helmet, vest, and/or mask is worn, and if a person is detected. The following features are included:

    - **Real-Time Detection Overlay:** Bounding boxes with class labels.
    - **Counts Display:** Real-time display of detected helmets, vests, masks, and persons.
    - **Email Alerts:** Alerts for any person without a helmet, with an image attachment of the frame.

---

## Upgrades in This Project

- **Email Alert with Attachment:** Sends an email every 10 seconds when a person is detected without a helmet, including the captured frame as an attachment.
- **Non-Blocking Email Process:** Email alerts are sent in the background to ensure smooth video streaming without interruptions.
- **Mail Sent Popup:** A popup notification is displayed in the top-right corner of the video feed to confirm that the email has been sent.
- **Real-Time Counting:** Real-time count of detected helmets, vests, masks, and persons is displayed on the sideboard.

---

## Customization

You can fine-tune the detection parameters, email settings, and thresholds in the `webcam.py` script to adapt to different environments and requirements.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This project is built upon the YOLOv8 architecture developed by [YOLO](https://github.com/AlexeyAB/darknet).
- Special thanks to the contributors and open-source community for their valuable insights and contributions.

# PPE Detection Application

This application uses computer vision to detect Personal Protective Equipment (PPE) violations in real-time using YOLO object detection.

## Prerequisites

- Docker
- Azure account with Blob Storage
- YOLO model file (`ppe.pt`) in the `Model` directory

## Environment Variables

Create a `.env` file with the following variables:
```
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_container_name
```

## Deployment Steps

1. Place your YOLO model file (`ppe.pt`) in the `Model` directory

2. Build the Docker image:
```bash
docker build -t ppe-detection-app .
```

3. Run the container:
```bash
docker run -p 5001:5001 --env-file .env ppe-detection-app
```

## Cloud Deployment Options

### Azure Container Instances (ACI)
```bash
# Login to Azure
az login

# Create a resource group
az group create --name ppe-detection-rg --location eastus

# Create the container instance
az container create \
  --resource-group ppe-detection-rg \
  --name ppe-detection \
  --image ppe-detection-app \
  --dns-name-label ppe-detection \
  --ports 5001 \
  --environment-variables AZURE_STORAGE_CONNECTION_STRING=your_connection_string AZURE_STORAGE_CONTAINER_NAME=your_container_name
```

### Azure App Service
1. Create an Azure Container Registry (ACR)
2. Push the Docker image to ACR
3. Create an App Service with container support
4. Configure environment variables
5. Deploy the container

### AWS Elastic Container Service (ECS)
1. Create an ECR repository
2. Push the Docker image to ECR
3. Create an ECS cluster
4. Create a task definition
5. Run the task with the container

## Accessing the Application

Once deployed, access the application at:
- Local: http://localhost:5001
- Cloud: http://your-deployment-url:5001

## Features

- Real-time PPE violation detection
- Video feed from camera or uploaded video
- Azure Blob Storage integration for violation images
- Real-time statistics and violation tracking
