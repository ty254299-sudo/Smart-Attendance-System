"""Recognize enrolled faces from a webcam and record daily attendance."""

import math
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np

from database import initialize_database, record_attendance

BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
LOG_DIR = BASE_DIR / "attendance_logs"
CAMERA_INDEX = 0
FRAME_SCALE = 0.25
MATCH_THRESHOLD = 0.55
CSV_COLUMNS = ["Roll_Name", "Date", "Timestamp"]
EYE_CLOSED_THRESHOLD = 0.21
EYE_OPEN_THRESHOLD = 0.25
MIN_CLOSED_FRAMES = 2


def load_known_faces() -> tuple[list, list[str]]:
    """Load the locally generated face encodings and their identifiers."""
    if not ENCODINGS_FILE.is_file():
        raise FileNotFoundError(
            f"Run 'encode_faces.py' first to generate {ENCODINGS_FILE.name}."
        )

    with ENCODINGS_FILE.open("rb") as encoding_file:
        data = pickle.load(encoding_file)

    encodings = data.get("encodings", [])
    names = data.get("names", [])
    if len(encodings) != len(names):
        raise ValueError("The encoding file has mismatched encodings and names.")
    return encodings, names


def mark_attendance(student_identifier: str) -> bool:
    """Record attendance once in SQLite and export the daily CSV for compatibility."""
    initialize_database()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"attendance_{today}.csv"
    timestamp = datetime.now().strftime("%H:%M:%S")
    added = record_attendance(student_identifier, today, timestamp)
    if not added:
        return False

    export_daily_csv(today)

    print(f"[LOGGED] Attendance marked for: {student_identifier} at {timestamp}")
    return True


def export_daily_csv(attendance_date: str) -> None:
    """Write the database's daily attendance table to the legacy CSV location."""
    import csv

    rows = fetch_attendance(attendance_date)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"attendance_{attendance_date}.csv"
    with log_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fetch_attendance(attendance_date: str | None = None) -> list[dict[str, str]]:
    """Get attendance rows from SQLite without exposing database internals."""
    from database import fetch_attendance as query_attendance

    initialize_database()
    return query_attendance(attendance_date)


def eye_aspect_ratio(eye: list[tuple[int, int]]) -> float:
    """Return the eye aspect ratio for six ordered facial landmark points."""
    if len(eye) != 6:
        raise ValueError("An eye must contain exactly six landmark points.")
    vertical_a = math.dist(eye[1], eye[5])
    vertical_b = math.dist(eye[2], eye[4])
    horizontal = math.dist(eye[0], eye[3])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class BlinkDetector:
    """Track a blink over consecutive frames for one face."""

    def __init__(self) -> None:
        self.closed_frames = 0
        self.verified = False

    def reset(self) -> None:
        """Require a new blink after the face changes or disappears."""
        self.closed_frames = 0
        self.verified = False

    def update(self, landmarks: dict[str, list[tuple[int, int]]]) -> bool:
        """Return true only on the frame that completes a blink."""
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])
        if len(left_eye) != 6 or len(right_eye) != 6:
            self.reset()
            return False

        ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        blink_detected = False
        if ratio < EYE_CLOSED_THRESHOLD:
            self.closed_frames += 1
        elif ratio > EYE_OPEN_THRESHOLD:
            if self.closed_frames >= MIN_CLOSED_FRAMES:
                self.verified = True
                blink_detected = True
            self.closed_frames = 0
        return blink_detected


def recognize_faces(frame: np.ndarray, known_encodings: list, known_names: list[str]):
    """Detect faces and return locations, labels, and eye landmarks."""
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_frame, locations)
    landmarks = face_recognition.face_landmarks(rgb_frame, locations)

    labels = []
    for face_encoding in encodings:
        label = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = int(np.argmin(distances))
            if distances[best_match_index] < MATCH_THRESHOLD:
                label = known_names[best_match_index]
        labels.append(label)

    return locations, labels, landmarks


def main() -> None:
    initialize_database()
    known_encodings, known_names = load_known_faces()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        video_capture.release()
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    face_locations = []
    face_names = []
    face_landmarks = []
    blink_detector = BlinkDetector()
    print("[INFO] Starting webcam. Press 'q' to exit.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[WARNING] Could not read a frame from the webcam.")
                break

            face_locations, face_names, face_landmarks = recognize_faces(
                frame, known_encodings, known_names
            )

            blink_detected = False
            if len(face_locations) == 1 and len(face_landmarks) == 1:
                blink_detected = blink_detector.update(face_landmarks[0])
            else:
                blink_detector.reset()
            liveness_verified = blink_detector.verified

            if blink_detected and len(face_names) == 1 and face_names[0] != "Unknown":
                mark_attendance(face_names[0])

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)
                verified_name = liveness_verified and name != "Unknown"
                color = (0, 255, 0) if verified_name else (0, 165, 255) if name != "Unknown" else (0, 0, 255)
                display_name = name if verified_name or name == "Unknown" else f"{name} - blink to verify"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    display_name,
                    (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
