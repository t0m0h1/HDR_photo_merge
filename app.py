import cv2
import numpy as np
from PIL import Image
import sys

def extract_exposure_times(image_paths):
    exposure_times = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                exif_data = img._getexif()
                exposure_time = float(exif_data[33434])  # Extracting exposure time from EXIF data
                exposure_times.append(exposure_time)
        except (AttributeError, KeyError, IndexError, FileNotFoundError) as e:
            print(f"Error processing image {path}: {e}")
            sys.exit(1)
    return np.array(exposure_times, dtype=np.float32)


def merge_images(images, exposure_times):
    try:
        merge_debvec = cv2.createMergeDebevec()
        hdr = merge_debvec.process(images, times=exposure_times)
        return hdr
    except cv2.error as e:
        print(f"Error merging images: {e}")
        sys.exit(1)



def tonemap(hdr, intensity=0.01):
    try:
        tonemap = cv2.createTonemapReinhard()
        tonemap.setIntensity(intensity)  # Adjust the intensity value (global exposure)
        ldr = tonemap.process(hdr)
        ldr = np.clip(ldr, 0, 1)
        return ldr
    except cv2.error as e:
        print(f"Error tonemapping HDR image: {e}")
        sys.exit(1)






if __name__ == "__main__":
    # Paths to input images
    image_paths = ["images/DJI_0965.JPG", "images/DJI_0966.JPG", "images/DJI_0967.JPG"]

    # Extract exposure times
    exposure_times = extract_exposure_times(image_paths)

    # Read input images
    images = [cv2.imread(path) for path in image_paths]

    # Merge images
    hdr = merge_images(images, exposure_times)

    # Tonemap HDR image (with adjusted intensity value for darkness)
    ldr = tonemap(hdr, intensity=0.1)  # Adjust intensity value as needed for much darker output

    # Save tonemapped image
    try:
        cv2.imwrite("output_hdr.jpg", (ldr * 255).astype(np.uint8))
    except cv2.error as e:
        print(f"Error saving tonemapped image: {e}")
        sys.exit(1)
