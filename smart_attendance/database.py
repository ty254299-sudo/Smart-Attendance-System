"""SQLite persistence layer for students and attendance records."""

import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "attendance.db"
LOG_DIR = BASE_DIR / "attendance_logs"


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transaction-managed SQLite connection and always close it."""
    connection = sqlite3.connect(DATABASE_FILE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    """Create the application schema and import old CSV records once safely."""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                identifier TEXT PRIMARY KEY,
                photo_filename TEXT,
                enrolled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_identifier TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(student_identifier) REFERENCES students(identifier),
                UNIQUE(student_identifier, attendance_date)
            );
            CREATE INDEX IF NOT EXISTS attendance_date_idx
                ON attendance(attendance_date);
            """
        )
        import_legacy_csvs(connection)


def import_legacy_csvs(connection: sqlite3.Connection) -> None:
    """Import existing daily CSV logs; the unique key makes this idempotent."""
    if not LOG_DIR.is_dir():
        return
    for log_file in LOG_DIR.glob("attendance_*.csv"):
        try:
            with log_file.open("r", newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    identifier = (row.get("Roll_Name") or "").strip()
                    date = (row.get("Date") or "").strip()
                    timestamp = (row.get("Timestamp") or "").strip()
                    if not identifier or not date or not timestamp:
                        continue
                    connection.execute(
                        "INSERT OR IGNORE INTO students(identifier) VALUES (?)",
                        (identifier,),
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO attendance
                           (student_identifier, attendance_date, timestamp)
                           VALUES (?, ?, ?)""",
                        (identifier, date, timestamp),
                    )
        except (OSError, UnicodeError, csv.Error):
            continue


def register_students(students: list[tuple[str, str]]) -> None:
    """Add or update several student photo records in one transaction."""
    with get_connection() as connection:
        connection.executemany(
            """INSERT INTO students(identifier, photo_filename)
               VALUES (?, ?)
               ON CONFLICT(identifier) DO UPDATE SET
                   photo_filename = excluded.photo_filename""",
            students,
        )


def record_attendance(identifier: str, attendance_date: str, timestamp: str) -> bool:
    """Record attendance once per student and date; return whether inserted."""
    with get_connection() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO students(identifier) VALUES (?)", (identifier,)
        )
        cursor = connection.execute(
            """INSERT OR IGNORE INTO attendance
               (student_identifier, attendance_date, timestamp)
               VALUES (?, ?, ?)""",
            (identifier, attendance_date, timestamp),
        )
        return cursor.rowcount == 1


def fetch_attendance(date: str | None = None) -> list[dict[str, str]]:
    """Fetch attendance records, newest first, optionally for one date."""
    query = """SELECT student_identifier AS Roll_Name,
                      attendance_date AS Date,
                      timestamp AS Timestamp
               FROM attendance"""
    parameters: tuple[str, ...] = ()
    if date:
        query += " WHERE attendance_date = ?"
        parameters = (date,)
    query += " ORDER BY attendance_date DESC, timestamp DESC, student_identifier"
    with get_connection() as connection:
        return [dict(row) for row in connection.execute(query, parameters)]


def fetch_students() -> list[dict[str, str | None]]:
    """Fetch enrolled students and their attendance totals."""
    with get_connection() as connection:
        rows = connection.execute(
            """SELECT s.identifier, s.photo_filename, s.enrolled_at,
                      COUNT(a.id) AS attendance_count
               FROM students AS s
               LEFT JOIN attendance AS a ON a.student_identifier = s.identifier
               WHERE s.photo_filename IS NOT NULL
               GROUP BY s.identifier, s.photo_filename, s.enrolled_at
               ORDER BY s.identifier"""
        )
        return [dict(row) for row in rows]


def fetch_dashboard_stats(today: str | None = None) -> dict[str, int]:
    """Return student, daily attendance, and overall attendance counts."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    with get_connection() as connection:
        student_count = connection.execute(
            "SELECT COUNT(*) FROM students WHERE photo_filename IS NOT NULL"
        ).fetchone()[0]
        present_today = connection.execute(
            "SELECT COUNT(*) FROM attendance WHERE attendance_date = ?", (today,)
        ).fetchone()[0]
        total_records = connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    return {
        "students": student_count,
        "present_today": present_today,
        "total_records": total_records,
    }
