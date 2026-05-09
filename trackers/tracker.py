from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import pandas as pd
from utils import get_center_of_bbox, get_width_of_bbox


class Tracker:
    # A referee track ID is "confirmed" once seen as referee this many times…
    REFEREE_CONFIRM_FRAMES = 5
    # …with at least this confidence each time
    REFEREE_CONF_THRESHOLD = 0.6

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.confirmed_referee_ids = set()   # track IDs permanently locked as referees
        self._referee_seen_count = {}        # {track_id: count of high-conf referee detections}

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += detection_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": [], "goalkeepers": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # NOTE: do NOT remap goalkeeper → player; keep them separate
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeepers"].append({})

            for frame_detection in detection_with_tracks:
                bbox       = frame_detection[0].tolist()
                confidence = float(frame_detection[2]) if frame_detection[2] is not None else 0.0
                cls_id     = frame_detection[3]
                track_id   = frame_detection[4]

                # --- Confirmed referee: always route to referees regardless of model class ---
                if track_id in self.confirmed_referee_ids:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    continue

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    # Accumulate high-confidence referee sightings
                    if confidence >= self.REFEREE_CONF_THRESHOLD:
                        self._referee_seen_count[track_id] = (
                            self._referee_seen_count.get(track_id, 0) + 1
                        )
                        if self._referee_seen_count[track_id] >= self.REFEREE_CONFIRM_FRAMES:
                            self.confirmed_referee_ids.add(track_id)

                elif "goalkeeper" in cls_names_inv and cls_id == cls_names_inv["goalkeeper"]:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """For ball drawing (ellipse at the bottom of the bbox)."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cv2.ellipse(
            frame,
            (x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )
        return frame

    def draw_rounded_rect(self, frame, x1, y1, x2, y2, color, radius=12, thickness=2, filled=False):
        """Draw a rounded rectangle using corner arcs and lines."""
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        fill_t = cv2.FILLED if filled else thickness

        if filled:
            # Fill as a solid shape using two overlapping filled rects + four corner circles
            cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, cv2.FILLED)
            cv2.circle(frame, (x1 + r, y1 + r), r, color, cv2.FILLED)
            cv2.circle(frame, (x2 - r, y1 + r), r, color, cv2.FILLED)
            cv2.circle(frame, (x1 + r, y2 - r), r, color, cv2.FILLED)
            cv2.circle(frame, (x2 - r, y2 - r), r, color, cv2.FILLED)
        else:
            # Top & bottom edges
            cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)
            # Left & right edges
            cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)
            # Corner arcs
            cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
            cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)

    def draw_player(self, frame, bbox, color, track_id):
        """Draw a rounded bounding box for a player with a track-ID badge at bottom-left."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # --- Rounded box outline ---
        self.draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=10, thickness=2)

        # --- Track-ID badge (bottom-left corner) ---
        label = str(track_id)
        font        = cv2.FONT_HERSHEY_DUPLEX
        font_scale  = 0.45
        font_thick  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        pad_x, pad_y = 6, 4
        bx1 = x1
        by2 = y2
        bx2 = x1 + tw + pad_x * 2
        by1 = y2 - th - pad_y * 2 - baseline

        # Filled badge background
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, color, radius=6, filled=True)
        # Badge border for crispness
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, (255, 255, 255), radius=6, thickness=1)

        # Track-ID text
        cv2.putText(
            frame, label,
            (bx1 + pad_x, by2 - baseline - pad_y),
            font, font_scale,
            (255, 255, 255),
            font_thick, cv2.LINE_AA,
        )
        return frame

    def draw_goalkeeper(self, frame, bbox, color):
        """Draw a rounded bounding box for a goalkeeper with a 'GK' badge at bottom-left."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # --- Rounded box outline (same style as player) ---
        self.draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=10, thickness=2)

        # --- 'GK' badge (bottom-left corner) ---
        label = "GK"
        font        = cv2.FONT_HERSHEY_DUPLEX
        font_scale  = 0.45
        font_thick  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        pad_x, pad_y = 6, 4
        bx1 = x1
        by2 = y2
        bx2 = x1 + tw + pad_x * 2
        by1 = y2 - th - pad_y * 2 - baseline

        # Filled badge background
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, color, radius=6, filled=True)
        # Badge border for crispness
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, (255, 255, 255), radius=6, thickness=1)

        # GK text
        cv2.putText(
            frame, label,
            (bx1 + pad_x, by2 - baseline - pad_y),
            font, font_scale,
            (255, 255, 255),
            font_thick, cv2.LINE_AA,
        )
        return frame

    def draw_referee(self, frame, bbox):
        """Draw a black rounded box for a referee with a 'Referee' label at bottom-left."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        box_color = (0, 0, 0)
        border_color = (0, 0, 0)

        # --- Rounded box outline ---
        self.draw_rounded_rect(frame, x1, y1, x2, y2, border_color, radius=10, thickness=2)

        # --- 'Referee' badge (bottom-left corner) ---
        label = "Referee"
        font        = cv2.FONT_HERSHEY_DUPLEX
        font_scale  = 0.38
        font_thick  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        pad_x, pad_y = 5, 3
        bx1 = x1
        by2 = y2
        bx2 = x1 + tw + pad_x * 2
        by1 = y2 - th - pad_y * 2 - baseline

        # Filled dark badge
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, box_color, radius=6, filled=True)
        self.draw_rounded_rect(frame, bx1, by1, bx2, by2, border_color, radius=6, thickness=1)

        cv2.putText(
            frame, label,
            (bx1 + pad_x, by2 - baseline - pad_y),
            font, font_scale,
            (255, 255, 255),
            font_thick, cv2.LINE_AA,
        )
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num in range(len(tracks["players"])):
            if frame_num >= len(video_frames):
                break
            frame = video_frames[frame_num].copy()

            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players — rounded box + bottom-left ID badge
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_player(frame, player["bbox"], color, track_id)

            # Draw Goalkeepers — green rounded box + 'GK' label
            GK_COLOR = (0, 200, 0)
            for _, goalkeeper in goalkeeper_dict.items():
                frame = self.draw_goalkeeper(frame, goalkeeper["bbox"], GK_COLOR)

            # Draw Referees — black rounded box + 'Referee' label
            for _, referee in referee_dict.items():
                frame = self.draw_referee(frame, referee["bbox"])

            # Draw Ball — keep original ellipse
            for _, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball["bbox"], (255, 255, 255))

            output_video_frames.append(frame)

        return output_video_frames
