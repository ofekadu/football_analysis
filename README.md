# ⚽ Football Analysis

Detect, track, and divide into groups the players, referees, and even the ball for any football clip using Machine Learning.

This project utilizes computer vision and machine learning techniques to analyze football match videos. It can identify individual players, referees, and the ball, track their movements across frames, and automatically categorize players into their respective teams based on their jersey colors.

---

## 🏗️ High-Level Architecture

The system processes football clips through a pipeline of computer vision and machine learning modules:

1. **Video Ingestion:** Reads the input football match video frame-by-frame.
2. **Object Detection & Tracking (`Tracker`):** Uses a fine-tuned YOLO model (`models/best.pt`) to detect and track entities in the frame:
   - Players
   - Referees
   - Ball
3. **Ball Position Interpolation:** Since the ball can be occluded or move too fast to be detected in every frame, the system interpolates missing ball positions to create a smooth trajectory.
4. **Team Assignment (`TeamAssigner`):** Analyzes the bounding boxes of the tracked players. It extracts the dominant colors from the players' jerseys using K-Means clustering technique and assigns each player to one of the two competing teams.
5. **Annotation & Rendering:** Draws ellipse below the objects and tracking IDs + team-specific colors directly onto the video frames.
6. **Video Export:** Compiles the annotated frames and saves the final output video.

---

## 💻 Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`)
- **Object Detection & Tracking:** YOLO (via `ultralytics`), `supervision`
- **Machine Learning & Data Processing:** `scikit-learn`, `numpy`, `pandas`

---

## 📁 Project Structure

```
football_analysis/
├── input_videos/          # Directory for input video clips
├── output_videos/         # Directory for annotated output videos
├── models/                # Contains ML models (e.g., best.pt - YOLO weights)
├── stubs/                 # Pickled tracker stubs for faster debugging/re-runs
├── trackers/              # Logic for object detection and tracking
├── team_assigner/         # Logic for color extraction and team assignment
├── utils/                 # Utility functions (video I/O, etc.)
├── main.py                # Main execution script
└── requirements.txt       # Python dependencies
```

---

## 🚀 Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd football_analysis
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model weights are present:**
   Place your trained YOLO weights file at `models/best.pt`.

---

## 🎮 Usage

1. Place your input football video in the `input_videos/` directory with the name `football_match.mp4`.
2. Run the main analysis script:
   ```bash
   python main.py
   ```
3. The processed video with annotations will be saved in the `output_videos/` directory with the name `output.avi`.

---

## 👨‍💻 Credits

- Created by: Ofek Adunsky | ofekadunsky@gmail.com
