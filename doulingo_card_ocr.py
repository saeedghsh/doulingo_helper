import argparse
import os
import glob
import cv2
import pytesseract
from typing import List

# For the pytesseract
# Set the path to the directory containing the language data files
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'


def find_largest_white_patch(image):
    """Return the bounding box of the biggest white patch.
    This white patch supposedly contains the text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    crop_bot = 60
    crop_top = 100
    xmin, xmax = x, x + w
    ymin, ymax = y + crop_top, y + h - crop_bot
    return xmin, ymin, xmax, ymax


def remove_trailing_blank_lines(text):
    lines = text.rstrip('\n').split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


def validate_dir_path(directory_path):
    """Return True if the path is a dir, it exists and is not empty."""
    if not os.path.exists(directory_path):
        return False
    if not os.path.isdir(directory_path):
        return False
    if not os.listdir(directory_path):
        return False
    return True


def image_to_text(image_path: str) -> str:
    """Return the text inside the image."""
    image = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = find_largest_white_patch(image)
    cropped_image = image[ymin:ymax, xmin:xmax]
    text = pytesseract.image_to_string(cropped_image, lang='swe')
    text = remove_trailing_blank_lines(text)
    text = text.replace("\n", "\t")
    return text


def all_texts_from_directory(
    directory_path: str, check_for_duplicates: List[str] = []
) -> List[str]:
    """Return a list of texts from all images in directory_path.

    If an image is a duplicate from the images in directory_path or the
    text already parsed in check_for_duplicates, delete that image and
    skip the text."""
    file_paths = glob.glob(os.path.join(directory_path, '*.png'))
    file_paths.sort()

    result = []
    for file_path in file_paths:
        text = image_to_text(file_path)
        if text not in result and text not in check_for_duplicates:
            result.append(text)
        else:
            print(f"this is a duplicate, deleting: {file_path}")
            os.remove(file_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directories",
        nargs="+",
        type=str,
        help="Path to the directories. The last one is the target and the preceeding are only to check for duplicates",
    )
    args = parser.parse_args()

    check_for_duplicates = []
    for directory in args.directories[:-1]:
        assert validate_dir_path(directory), f"bad directory path {directory}"
        check_for_duplicates.extend(all_texts_from_directory(directory, check_for_duplicates))

    assert validate_dir_path(args.directories[-1]), f"bad directory path {args.directories[-1]}"
    result = all_texts_from_directory(args.directories[-1], check_for_duplicates)

    with open(f"{args.directories[-1]}/result.txt", 'a') as file:
        file.write("\n".join(result))
