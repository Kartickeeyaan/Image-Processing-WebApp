# utils/image_io.py
import os
import uuid
import cv2

ALLOWED_EXTS = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def unique_name(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"

def save_image(path: str, img) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def read_bgr(path: str):
    return cv2.imread(path)  # returns BGR or None
