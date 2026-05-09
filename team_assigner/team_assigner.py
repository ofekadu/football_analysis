import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    VOTE_FRAMES = 10  # per-player detections before locking in team

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}   # locked-in assignments (after VOTE_FRAMES votes)
        self.player_team_votes = {}  # {player_id: [team_id, team_id, ...]}

    # ------------------------------------------------------------------
    # Color extraction helpers
    # ------------------------------------------------------------------

    def _get_jersey_crop(self, frame, bbox):
        """Return a tight crop focused on the jersey (upper center of bbox).

        Using only the upper ~40 % of the box and trimming 15 % from each
        side reduces background, legs, and arm pixels significantly.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = y2 - y1, x2 - x1

        cx1 = x1 + int(w * 0.15)
        cx2 = x2 - int(w * 0.15)
        cy1 = y1 + int(h * 0.05)
        cy2 = y1 + int(h * 0.45)

        crop = frame[max(cy1, 0):max(cy2, 0), max(cx1, 0):max(cx2, 0)]
        if crop.size == 0:
            # Fallback: full bbox
            crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        return crop

    def _dominant_color(self, crop):
        """Extract the dominant jersey color after filtering out noise pixels.

        Filters removed:
          - Pitch green  (HSV hue 35-85, saturation > 50)
          - Very dark    (value < 30)
          - Washed-out   (saturation < 15 and value > 200)  — white lines etc.

        Then runs KMeans(k=2) and returns the center of the largest cluster,
        which is typically the jersey rather than a background remnant.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].ravel().astype(np.int32)
        s_ch = hsv[:, :, 1].ravel().astype(np.int32)
        v_ch = hsv[:, :, 2].ravel().astype(np.int32)
        pixels = crop.reshape(-1, 3).astype(np.float32)

        is_green  = (h_ch >= 35) & (h_ch <= 85) & (s_ch > 50)
        is_dark   = v_ch < 30
        is_washed = (s_ch < 15) & (v_ch > 200)
        valid = ~(is_green | is_dark | is_washed)

        filtered = pixels[valid]
        if len(filtered) < 20:
            filtered = pixels  # fallback: use all pixels

        km = KMeans(n_clusters=2, init="k-means++", n_init=3).fit(filtered)
        counts = np.bincount(km.labels_)
        # Dominant cluster = most pixels = jersey; return as float32 for sklearn compatibility
        return km.cluster_centers_[np.argmax(counts)].astype(np.float32)

    def get_player_color(self, frame, bbox):
        crop = self._get_jersey_crop(frame, bbox)
        return self._dominant_color(crop)

    # ------------------------------------------------------------------
    # Team color calibration
    # ------------------------------------------------------------------

    def assign_team_color(self, frame_player_pairs):
        """Fit team color KMeans from multiple (frame, player_dict) pairs.

        Args:
            frame_player_pairs: list of (frame, player_detections_dict) tuples.
        """
        player_colors = []
        for frame, player_detections in frame_player_pairs:
            for _, player_detection in player_detections.items():
                color = self.get_player_color(frame, player_detection["bbox"])
                player_colors.append(color)

        if not player_colors:
            return

        # Fit on a consistent float32 array to avoid sklearn dtype mismatch
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(
            np.array(player_colors, dtype=np.float32)
        )
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    # ------------------------------------------------------------------
    # Per-player team assignment (voting)
    # ------------------------------------------------------------------

    def get_player_team(self, frame, player_bbox, player_id):
        # Already locked in — return immediately
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Predict team for this frame
        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1).astype(np.float32))[0]) + 1

        # Accumulate vote
        if player_id not in self.player_team_votes:
            self.player_team_votes[player_id] = []
        self.player_team_votes[player_id].append(team_id)

        # Lock in once we have enough votes (majority wins)
        votes = self.player_team_votes[player_id]
        if len(votes) >= self.VOTE_FRAMES:
            locked_team = max(set(votes), key=votes.count)
            self.player_team_dict[player_id] = locked_team
            return locked_team

        # Still collecting — return current majority so display is never empty
        return max(set(votes), key=votes.count)
