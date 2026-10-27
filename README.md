# Smart Attendance System

## Setup

Use a Python environment supported by the existing `face-recognition`/`dlib`
dependencies. On Windows, installing `dlib` may require additional build tools;
follow the installation instructions for your Python version if pip cannot find
a compatible wheel.

From the `smart_attendance` directory, install the packages listed in
`requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Admin dashboard

Start the browser dashboard from the `smart_attendance` directory:

```powershell
streamlit run admin_dashboard.py
```

The Streamlit app is the browser-facing frontend, its Python modules provide the
application/backend logic, and `attendance.db` is the SQLite database created
automatically in `smart_attendance/`. It stores student profiles and one
attendance row per student per day. Existing daily CSV files are imported into
SQLite when the app starts; newly recorded attendance is also exported to
`attendance_logs/` for compatibility. The dashboard includes attendance summary
cards, enrollment, live verification, date-filtered CSV downloads, and a student
roster with attendance totals.

In **Student enrollment**, upload one clear image per student. Use filenames in
the format `RollNumber_Name.jpg` (or `.jpeg` / `.png`). The dashboard checks that
each image contains exactly one detectable face, saves the images in `dataset/`,
and rebuilds the face encodings. In **Live attendance**, allow browser camera
access, keep just one person in frame, and blink. The app records a recognized
student to the SQLite database only after a blink is detected. The **Attendance
logs** tab displays database records and allows the filtered table to be
downloaded as CSV.

The blink check is a simple eye-landmark liveness heuristic; it is not a
security-grade anti-spoofing guarantee. Good lighting and a clear, forward-facing
view improve detection. The WebRTC dashboard requests video from the browser,
so camera permission is required and deployments may need suitable WebRTC/STUN
network configuration.

## Desktop camera mode

The original desktop interface is still available with:

```powershell
python attendance_system.py
```

It also requires one face to blink before logging attendance.