import os

from PIL import Image


def clean_data(root: str):
    for root, dirs, files in os.walk(root):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                print(f"Deleting corrupted image: {path}")
                os.remove(path)
