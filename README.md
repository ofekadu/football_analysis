# ⚽ Football Analysis

Detect, track, and divide into groups the players, referees, goalkeepers, and the ball for any football clip using Machine Learning.

This project utilizes computer vision and machine learning techniques to analyze football match videos. It identifies individual players, referees, and goalkeepers, tracks their movements across frames, and automatically categorizes players into their respective teams based on jersey colors.

---

## 🎥 Demo

Check out this [video showcasing the evolution of the project](https://drive.google.com/file/d/1dfEDdLNwOL2SBUaZjs37T2XJS5gZhuwl/view?usp=sharing).

---

## 🏗️ High-Level Architecture

The system processes football clips through a pipeline of computer vision and machine learning modules:

1. **Video Ingestion:** Reads the input football match video frame-by-frame.
2. **Object Detection & Tracking (`Tracker`):** Uses a fine-tuned YOLO model (`models/best.pt`) to detect and track entities:
   - **Players** — each gets a team-colored rounded bounding box with their tracking number badge.
   - **Goalkeepers** — tracked as a separate class (not merged with players); rendered with a fixed green rounded box and a **GK** badge.
   - **Referees** — rendered with a black rounded box and a **Referee** label. Once a track ID is seen as a referee with high confidence for 5 consecutive frames it is permanently locked in, preventing the model from later mis-classifying that person as a player.
   - **Ball** — tracked with a white ellipse at its base.
3. **Ball Position Interpolation:** Missing ball positions are interpolated frame-to-frame to create a smooth trajectory even through occlusions.
4. **Team Assignment (`TeamAssigner`):** Multi-stage jersey-color clustering pipeline:
   - **Tight jersey crop** — uses only the upper-center region of each bounding box (top 40 %, inner 70 % width) to minimize background contamination.
   - **Green-pixel filtering** — HSV-based mask removes pitch grass, very dark pixels, and washed-out pixels before clustering.
   - **Dominant-cluster selection** — KMeans(k=2) is run on the filtered pixels; the _largest_ cluster is taken as the jersey color (replaces the fragile corner-pixel heuristic).
   - **Multi-frame calibration** — team color centroids are fitted on the first 10 frames (≈160 player crops) rather than a single frame, giving a far more robust baseline.
   - **Per-player voting** — each player's team prediction is accumulated over 10 detections before being locked in via majority vote, preventing a single bad crop from sticking for the whole video.
5. **Annotation & Rendering:** Custom drawing pipeline using OpenCV:
   - Rounded bounding boxes (built from line segments + arc corners) for all entity types.
   - Per-entity badge system: track-ID badge (players), **GK** badge (goalkeepers), **Referee** badge (referees).
6. **Video Export:** Compiles annotated frames and saves the final output video.

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
├── trackers/              # Object detection, tracking, and annotation logic
├── team_assigner/         # Jersey color extraction and team assignment
├── utils/                 # Utility functions (video I/O, bounding-box helpers)
├── main.py                # Main execution script
└── requirements.txt       # Python dependencies
```

---

## 🚀 Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ofekadu/football_analysis.git
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
