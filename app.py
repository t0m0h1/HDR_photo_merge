import cv2
import numpy as np
from PIL import Image
import sys
import logging

EXIF_TAG_EXPOSURE_TIME = 33434
IMAGE_PATHS = ["images/img1.JPG", "images/img2.JPG", "images/img3.JPG"]
OUTPUT_IMAGE_PATH = "output_hdr.jpg"

logging.basicConfig(level=logging.INFO)

def extract_exposure_times(image_paths):
    """Extract exposure times from images."""
    exposure_times = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                exif_data = img._getexif()
                exposure_time = exif_data.get(EXIF_TAG_EXPOSURE_TIME)
                if exposure_time is not None:
                    exposure_times.append(float(exposure_time))
                else:
                    logging.warning(f"Exposure time not found in image {path}")
        except (AttributeError, KeyError, IndexError, FileNotFoundError) as e:
            logging.error(f"Error processing image {path}: {e}")
            raise
    if len(exposure_times) < 2:
        raise ValueError("Insufficient input images with exposure time information.")
    return np.array(exposure_times, dtype=np.float32)

def merge_images(images):
    """Merge images into HDR."""
    try:
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(images)
        return hdr
    except cv2.error as e:
        logging.error(f"Error merging images: {e}")
        raise

def tonemap(hdr, tonemap_algorithm=cv2.createTonemapReinhard, parameters=None):
    """Tonemap HDR image."""
    try:
        if parameters is None:
            parameters = {}
        tonemapper = tonemap_algorithm(**parameters)
        ldr = tonemapper.process(hdr)
        return ldr
    except cv2.error as e:
        logging.error(f"Error tonemapping HDR image: {e}")
        raise

if __name__ == "__main__":
    exposure_times = extract_exposure_times(IMAGE_PATHS)

    images = []
    for path in IMAGE_PATHS:
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Error loading image {path}")
            sys.exit(1)
        images.append(img)

    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 1 else img for img in images]

    logging.info("Shape of loaded images:")
    for img in images:
        logging.info(img.shape)

    hdr = merge_images(images)

    tonemap_parameters = {
        'gamma': 3.0,     # Increase gamma for more contrast
        'intensity': 0.1, # Decrease intensity for a darker output
    }

    ldr = tonemap(hdr, parameters=tonemap_parameters)

    try:
        cv2.imwrite(OUTPUT_IMAGE_PATH, (ldr * 255).astype(np.uint8))
        logging.info("HDR merge completed successfully.")
    except cv2.error as e:
        logging.error(f"Error saving tonemapped image: {e}")
        sys.exit(1)