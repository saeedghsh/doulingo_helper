import argparse
import os
import glob
import cv2
import pytesseract

# For the pytesseract
# Set the path to the directory containing the language data files
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'  # Replace with the actual path


def find_largest_white_patch(image):
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
    if not os.path.exists(directory_path):
        return False
    if not os.path.isdir(directory_path):
        return False
    if not os.listdir(directory_path):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to the directory")
    args = parser.parse_args()
    assert validate_dir_path(args.directory), "bad directory path"

    file_paths = glob.glob(os.path.join(args.directory, '*.png'))

    result = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        xmin, ymin, xmax, ymax = find_largest_white_patch(image)
        cropped_image = image[ymin:ymax, xmin:xmax]
        text = pytesseract.image_to_string(cropped_image, lang='swe')
        text = remove_trailing_blank_lines(text)
        text = text.replace("\n", "\t")
        result.append(text)

    with open(f"{args.directory}/result.txt", 'a') as file:
        file.write("---\n".join(result))
