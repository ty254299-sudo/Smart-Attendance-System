"""Recognize enrolled faces from a webcam and record daily attendance."""

import csv
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
LOG_DIR = BASE_DIR / "attendance_logs"
CAMERA_INDEX = 0
FRAME_SCALE = 0.25
FRAME_SKIP = 3
MATCH_THRESHOLD = 0.55
CSV_COLUMNS = ["Roll_Name", "Date", "Timestamp"]


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
    """Add a student to today's CSV once; return whether a row was added."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"attendance_{today}.csv"

    if log_file.exists():
        with log_file.open("r", newline="", encoding="utf-8") as csv_file:
            existing_rows = csv.DictReader(csv_file)
            if any(row.get("Roll_Name") == student_identifier for row in existing_rows):
                return False

    timestamp = datetime.now().strftime("%H:%M:%S")
    write_header = not log_file.exists() or log_file.stat().st_size == 0
    with log_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {"Roll_Name": student_identifier, "Date": today, "Timestamp": timestamp}
        )

    print(f"[LOGGED] Attendance marked for: {student_identifier} at {timestamp}")
    return True


def recognize_faces(frame: np.ndarray, known_encodings: list, known_names: list[str]):
    """Detect faces in a frame and return locations with recognized labels."""
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    labels = []
    for face_encoding in encodings:
        label = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = int(np.argmin(distances))
            if distances[best_match_index] < MATCH_THRESHOLD:
                label = known_names[best_match_index]
                mark_attendance(label)
        labels.append(label)

    return locations, labels


def main() -> None:
    known_encodings, known_names = load_known_faces()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    if not video_capture.isOpened():
        video_capture.release()
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    face_locations = []
    face_names = []
    frame_count = 0
    print("[INFO] Starting webcam. Press 'q' to exit.")

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("[WARNING] Could not read a frame from the webcam.")
                break

            if frame_count % FRAME_SKIP == 0:
                face_locations, face_names = recognize_faces(
                    frame, known_encodings, known_names
                )
            frame_count += 1

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    name,
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
