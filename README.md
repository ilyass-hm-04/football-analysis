# Football Analysis Project

This project is a comprehensive computer vision system designed to analyze football match footage. It leverages state-of-the-art object detection and tracking techniques to extract meaningful insights from video data, such as player movement, ball possession, and team strategies.

## Features

- **Object Detection & Tracking**: Utilizes YOLO (You Only Look Once) to detect and track players, referees, and the ball across video frames.
- **Team Assignment**: Automatically assigns players to teams based on jersey colors using color clustering techniques.
- **Ball Acquisition**: Determines which player is in possession of the ball at any given time.
- **Camera Movement Estimation**: Compensates for camera motion to provide accurate player tracking and analysis relative to the field.
- **View Transformation**: Transforms the perspective of the video to a top-down 2D view of the field for tactical analysis.
- **Speed and Distance Estimation**: Calculates the speed and distance covered by each player throughout the match.

## Project Structure

- `main.py`: The main entry point of the application. Orchestrates the entire analysis pipeline.
- `yolo_inference.py`: A script for running YOLO inference on video files.
- `camera_movement_estimateor/`: Module for estimating and compensating for camera movement.
- `player_ball_assigner/`: Module for assigning ball possession to players.
- `speed_and_distance_estimator/`: Module for calculating player speed and distance covered.
- `team_assigner/`: Module for assigning detected players to their respective teams.
- `trackers/`: Module for object tracking and handling track data.
- `view_transformer/`: Module for transforming video coordinates to real-world field coordinates.
- `utils/`: Utility functions for video I/O and other helper tasks.
- `models/`: Directory for storage of trained YOLO models (e.g., `best.pt`, `yolov8s.pt`).
- `input_videos/`: Directory for input video files.
- `output_videos/`: Directory for saving the processed output videos.
- `stubs/`: Directory for storing intermediate data (stubs) to speed up development and testing.

## Prerequisites

- Python 3.x
- Ultralytics YOLO
- OpenCV (`cv2`)
- NumPy

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/football-analysis.git
    cd football-analysis
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Input Video**: Place your football match video in the `input_videos` directory. The default code expects a file named `08fd33_4.mp4` (you may need to update the path in `main.py`).

2.  **Download Models**: Ensure you have the necessary YOLO model weights. The code expects `models/best.pt` or `models/last.pt`. You can also use standard YOLO weights like `yolov8s.pt`.

3.  **Run the Analysis**:
    To run the full analysis pipeline, execute the `main.py` script:
    ```bash
    python main.py
    ```

    To run just the YOLO inference test:
    ```bash
    python yolo_inference.py
    ```

4.  **View Results**: The processed video with annotations (tracks, team colors, speed, distance) will be saved in the `output_videos` directory (e.g., `output_videos/output_video.avi`).

## Customization

- **Model Paths**: Update the model paths in `main.py` and `yolo_inference.py` if your model weights are located elsewhere.
- **Input Video**: Change the video file path in `read_video()` within `main.py` to analyze different videos.
- **Stubs**: The system uses pickle files in `stubs/` to cache tracking and camera movement data. Set `read_from_stub=False` in `main.py` when running on a new video to force re-computation.
