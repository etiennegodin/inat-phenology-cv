import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)


def clean_data(image_dir: str):
    for image_dir, dirs, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(image_dir, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                logger.info(f"Deleting corrupted image: {path}")
                os.remove(path)
