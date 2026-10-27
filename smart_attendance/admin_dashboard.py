"""Streamlit administration dashboard for student enrollment and attendance."""

import re
import threading
from pathlib import Path

import av
import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from attendance_system import (
    BlinkDetector,
    FRAME_SCALE,
    LOG_DIR,
    load_known_faces,
    mark_attendance,
    recognize_faces,
)
from database import (
    fetch_attendance as query_attendance,
    fetch_dashboard_stats,
    fetch_students,
    initialize_database,
)
from encode_faces import DATASET_DIR, ENCODINGS_FILE, encode_known_faces

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class AttendanceVideoProcessor(VideoProcessorBase):
    """Recognize one face and log it only after a live blink is observed."""

    def __init__(self) -> None:
        self.known_encodings, self.known_names = load_known_faces()
        self.blink_detector = BlinkDetector()
        self.log_lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        locations, labels, landmarks = recognize_faces(
            image, self.known_encodings, self.known_names
        )

        blink_detected = False
        if len(locations) == 1 and len(landmarks) == 1:
            blink_detected = self.blink_detector.update(landmarks[0])
        else:
            self.blink_detector.reset()
        verified = self.blink_detector.verified

        for (top, right, bottom, left), label in zip(locations, labels):
            top, right, bottom, left = (
                int(top / FRAME_SCALE),
                int(right / FRAME_SCALE),
                int(bottom / FRAME_SCALE),
                int(left / FRAME_SCALE),
            )
            known = label != "Unknown"
            if known and blink_detected:
                with self.log_lock:
                    mark_attendance(label)
            if known and verified:
                display_label = f"{label} - verified"
                color = (0, 180, 0)
            elif known:
                display_label = f"{label} - blink to verify"
                color = (0, 165, 255)
            else:
                display_label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(
                image,
                display_label,
                (left, max(24, top - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                color,
                2,
            )

        if len(locations) > 1:
            cv2.putText(
                image,
                "Only one person should be in frame",
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        elif len(locations) == 0:
            cv2.putText(
                image,
                "Position your face in view, then blink",
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def load_attendance_logs(date: str | None = None) -> pd.DataFrame:
    """Load attendance records from the SQLite persistence layer."""
    return pd.DataFrame(
        query_attendance(date), columns=["Roll_Name", "Date", "Timestamp"]
    )


def apply_dashboard_style() -> None:
    """Apply a compact admin-dashboard visual theme."""
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb; }
        [data-testid="stSidebar"] { background: #101b35; }
        [data-testid="stSidebar"] * { color: #e8eefb; }
        [data-testid="stSidebar"] [data-testid="stAlert"] * { color: #17243b; }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.65rem 1.9rem; border-radius: 20px; color: white;
                background: linear-gradient(120deg, #172b50 0%, #2955a3 58%, #3979d6 100%);
                box-shadow: 0 16px 38px rgba(27, 57, 103, .18); margin-bottom: 1.2rem; }
        .hero-kicker { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem;
                       opacity: .76; font-weight: 700; }
        .hero h1 { color: white; margin: .35rem 0 .25rem; font-size: 2rem; }
        .hero p { color: #e5edfc; margin: 0; }
        [data-testid="stMetric"] { background: white; padding: 1rem 1.15rem;
            border: 1px solid #e6ebf3; border-radius: 16px;
            box-shadow: 0 5px 16px rgba(28, 48, 80, .045); }
        [data-testid="stMetricLabel"] { color: #62708a; }
        [data-testid="stMetricValue"] { color: #162642; }
        .section-note { color: #6b7890; margin-top: -.5rem; margin-bottom: 1rem; }
        div[data-testid="stTabs"] button { border-radius: 9px 9px 0 0; }
        div[data-testid="stDataFrame"] { border: 1px solid #e5ebf3; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_enrollment() -> None:
    st.subheader("Enroll students")
    st.caption(
        "Upload one clear, front-facing photo per student. Name each file "
        "RollNumber_Name (for example, 1042_Alex.jpg)."
    )
    uploads = st.file_uploader(
        "Student photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="student_photos",
    )
    if st.button("Save photos and rebuild face encodings", disabled=not uploads):
        accepted = []
        seen_identifiers = set()
        issues = []
        for upload in uploads or []:
            suffix = Path(upload.name).suffix.lower()
            identifier = Path(upload.name).stem
            if suffix not in IMAGE_EXTENSIONS or not IDENTIFIER_PATTERN.fullmatch(identifier):
                issues.append(f"{upload.name}: use a simple RollNumber_Name filename.")
                continue
            if identifier in seen_identifiers:
                issues.append(f"{upload.name}: duplicate student identifier in this upload.")
                continue

            image_bytes = upload.getvalue()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                issues.append(f"{upload.name}: could not read this image.")
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if len(face_locations) != 1 or len(face_encodings) != 1:
                issues.append(f"{upload.name}: expected exactly one detectable face.")
                continue

            seen_identifiers.add(identifier)
            accepted.append((DATASET_DIR / f"{identifier}{suffix}", image_bytes))

        for issue in issues:
            st.error(issue)
        if accepted:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            for destination, image_bytes in accepted:
                destination.write_bytes(image_bytes)
            try:
                encode_known_faces()
                initialize_database()
                st.success(f"Saved {len(accepted)} photo(s) and rebuilt face encodings.")
                st.rerun()
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Photos were saved, but encoding failed: {error}")
        elif not issues:
            st.info("Choose at least one photo to enroll.")


def render_attendance() -> None:
    st.subheader("Live attendance")
    if not ENCODINGS_FILE.is_file():
        st.warning("Enroll at least one student before starting the camera.")
        return

    st.caption(
        "Allow camera access, keep one face in view, and blink naturally. "
        "Attendance is recorded only after a blink is detected."
    )
    webrtc_streamer(
        key="attendance-camera",
        video_processor_factory=AttendanceVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


def render_logs() -> None:
    st.subheader("Attendance logs")
    st.markdown('<p class="section-note">Filter, review, and export attendance stored in SQLite.</p>', unsafe_allow_html=True)
    if st.button("Refresh logs"):
        st.rerun()

    all_logs = load_attendance_logs()
    if all_logs.empty:
        st.info("No attendance entries have been recorded yet.")
        return

    dates = sorted(all_logs["Date"].dropna().astype(str).unique(), reverse=True)
    selected_date = st.selectbox("Filter by date", ["All dates", *dates])
    visible_logs = (
        all_logs
        if selected_date == "All dates"
        else load_attendance_logs(selected_date)
    )

    st.metric("Records", len(visible_logs))
    st.dataframe(visible_logs, use_container_width=True, hide_index=True)
    csv_bytes = visible_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download displayed logs as CSV",
        data=csv_bytes,
        file_name="attendance_logs.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Smart Attendance Admin", page_icon="📋", layout="wide")
    initialize_database()
    apply_dashboard_style()
    st.markdown(
        """<section class="hero"><div class="hero-kicker">Attendance management</div>
        <h1>Smart Attendance</h1><p>Your student roster, live verification, and daily records in one place.</p></section>""",
        unsafe_allow_html=True,
    )

    stats = fetch_dashboard_stats()
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Enrolled students", stats["students"])
    metric_two.metric("Present today", stats["present_today"])
    metric_three.metric("Attendance records", stats["total_records"])

    with st.sidebar:
        st.markdown("## Admin panel")
        st.caption("LOCAL ATTENDANCE SYSTEM")
        st.divider()
        st.write(f"**Face encodings**  ·  {'Ready' if ENCODINGS_FILE.is_file() else 'Not generated'}")
        st.write(f"**Database**  ·  SQLite")
        st.write(f"**CSV export folder**  ·  `{LOG_DIR.name}/`")
        st.divider()
        st.info(
            "Blink detection is a basic liveness check, not a guarantee against "
            "all spoofing attacks."
        )

    enrollment_tab, attendance_tab, logs_tab, students_tab = st.tabs(
        ["＋  Enrollment", "◉  Live attendance", "▤  Attendance logs", "♙  Students"]
    )
    with enrollment_tab:
        render_enrollment()
    with attendance_tab:
        render_attendance()
    with logs_tab:
        render_logs()
    with students_tab:
        st.subheader("Student roster")
        students = pd.DataFrame(fetch_students())
        if students.empty:
            st.info("No enrolled students yet. Add student photos in Enrollment.")
        else:
            st.markdown('<p class="section-note">Registered profiles and attendance totals.</p>', unsafe_allow_html=True)
            st.dataframe(students, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()