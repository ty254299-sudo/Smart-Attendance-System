"""Create face encodings from images in the dataset directory."""

import pickle
from pathlib import Path

import face_recognition

from database import initialize_database, register_students

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def encode_known_faces() -> None:
    """Encode images named <RollNumber>_<Name> and save them locally."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    image_paths = sorted(
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"[WARNING] No supported images found in '{DATASET_DIR}'.")
        return

    known_encodings = []
    known_names = []
    enrolled_students = []

    print("[INFO] Processing dataset images...")
    for image_path in image_paths:
        identifier = image_path.stem
        image = face_recognition.load_image_file(str(image_path))
        locations = face_recognition.face_locations(image, model="hog")

        if len(locations) != 1:
            print(
                f"  [-] Skipping {image_path.name}: expected exactly one face, "
                f"found {len(locations)}."
            )
            continue

        encodings = face_recognition.face_encodings(image, locations)
        if not encodings:
            print(f"  [-] Warning: Could not encode a face in {image_path.name}")
            continue

        known_encodings.append(encodings[0])
        known_names.append(identifier)
        enrolled_students.append((identifier, image_path.name))
        print(f"  [+] Enrolled: {identifier}")

    data = {"encodings": known_encodings, "names": known_names}
    with ENCODINGS_FILE.open("wb") as encoding_file:
        pickle.dump(data, encoding_file)
    initialize_database()
    register_students(enrolled_students)

    print(f"\n[SUCCESS] Encoded {len(known_names)} face(s) to '{ENCODINGS_FILE.name}'.")


if __name__ == "__main__":
    encode_known_faces()
