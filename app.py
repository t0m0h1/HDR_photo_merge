import cv2
import numpy as np
from PIL import Image

def extract_exposure_times(image_paths):
    exposure_times = []
    for path in image_paths:
        with Image.open(path) as img:
            exif_data = img._getexif()
            exposure_time = exif_data[33434] / exif_data[33437]  # Extracting exposure time from EXIF data
            exposure_times.append(exposure_time)
    return np.array(exposure_times, dtype=np.float32)

def merge_images(images, exposure_times):
    merge_debvec = cv2.createMergeDebevec()
    hdr = merge_debvec.process(images, times=exposure_times)
    return hdr

def tonemap(hdr):
    tonemap = cv2.createTonemapDurand(gamma=2.2)
    ldr = tonemap.process(hdr)
    ldr = 3 * ldr
    return np.clip(ldr, 0, 1)

if __name__ == "__main__":
    # Paths to input images
    image_paths = ["../images/DJI0965.JPG", "/images/DJI0966.JPG", "/images/DJI0967.JPG"]

    # Extract exposure times
    exposure_times = extract_exposure_times(image_paths)

    # Read input images
    images = [cv2.imread(path) for path in image_paths]

    # Merge images
    hdr = merge_images(images, exposure_times)

    # Tonemap HDR image
    ldr = tonemap(hdr)

    # Save tonemapped image
    cv2.imwrite("output_hdr.jpg", np.clip(ldr * 255, 0, 255).astype(np.uint8))
