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
                exposure_time = exif_data.get(33434)  # EXIF tag for exposure time
                if exposure_time is not None:
                    exposure_times.append(float(exposure_time))
                else:
                    print(f"Exposure time not found in image {path}")
        except (AttributeError, KeyError, IndexError, FileNotFoundError) as e:
            print(f"Error processing image {path}: {e}")
            sys.exit(1)
    if len(exposure_times) < 2:
        print("Insufficient input images with exposure time information.")
        sys.exit(1)
    return np.array(exposure_times, dtype=np.float32)


def merge_images(images):
    try:
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(images)
        return hdr
    except cv2.error as e:
        print(f"Error merging images: {e}")
        sys.exit(1)


def tonemap(hdr, tonemap_algorithm=cv2.TonemapMantiuk, parameters=None):
    try:
        if parameters is None:
            parameters = {}
        tonemapper = tonemap_algorithm(**parameters)
        ldr = tonemapper.process(hdr)
        return ldr
    except cv2.error as e:
        print(f"Error tonemapping HDR image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    image_paths = ["images/img1.JPG", "images/img2.JPG", "images/img3.JPG"]

    exposure_times = extract_exposure_times(image_paths)

    images = [cv2.imread(path) for path in image_paths]
    if any(img is None for img in images):
        print("Error loading one or more input images.")
        sys.exit(1)

    # Convert images to RGB format if needed
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 1 else img for img in images]

    print("Shape of loaded images:")
    for img in images:
        print(img.shape)

    hdr = merge_images(images)
    if hdr is None:
        print("Error merging images. Check input images and exposure times.")
        sys.exit(1)

    tonemap_parameters = {
        'gamma': 1.0,     # Adjust gamma for brightness
        'saturation': 1.0 # Adjust saturation
    }

    ldr = tonemap(hdr, tonemap_algorithm=cv2.createTonemapMantiuk, parameters=tonemap_parameters)
    if ldr is None:
        print("Error tonemapping HDR image.")
        sys.exit(1)

    try:
        cv2.imwrite("output_hdr.jpg", (ldr * 255).astype(np.uint8))
        print("HDR merge completed successfully.")
    except cv2.error as e:
        print(f"Error saving tonemapped image: {e}")
        sys.exit(1)
