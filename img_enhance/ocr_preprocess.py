"""
    Adapted from https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/
    Modified by Eva Natinsky (evamn97), April 2023

    Image preprocessing for OCR

    Usage:  python ocr_preprocess.py -i results
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
import glob


# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", default="results/", help="path to input images to be OCR'd")
    args = parser.parse_args()
    return args


def im_preprocess(image_path):

    # load the input image and convert it to grayscale
    image = cv2.imread(image_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image_gray, cmap='gray')
    plt.title("grayscale input image")
    plt.tight_layout()
    plt.show()

    # threshold the image using Otsu's thresholding method
    image_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # plt.imshow(image_thresh, cmap='gray')
    # plt.title("threshold image")
    # plt.tight_layout()
    # plt.show()

    # ** the following distance transforms work best on high-resolution images; on low-res it tends to reject letters as well
    # apply a distance transform which calculates the distance to the
    # closest zero pixel for each pixel in the input image
    # image_dist = cv2.distanceTransform(image_thresh, cv2.DIST_L1, 5)
    #
    # normalize the distance transform such that the distances lie in
    # the range [0, 1] and then convert the distance transform back to
    # an unsigned 8-bit integer in the range [0, 255]
    # image_dist = cv2.normalize(image_dist, image_dist, 0, 1.0, cv2.NORM_MINMAX)
    # image_dist = (image_dist * 255).astype("uint8")
    # plt.imshow(image_dist, cmap='gray')
    # plt.title("distance transform")
    # plt.tight_layout()
    # plt.show()
    #
    # threshold the distance transform using Otsu's method
    # image_dist_thresh = cv2.threshold(image_dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # plt.imshow(image_dist_thresh, cmap='gray')
    # plt.title("distance transform threshold")
    # plt.tight_layout()
    # plt.show()

    # apply an "opening" morphological operation to disconnect components
    # in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
    # plt.imshow(image_opening, cmap='gray')
    # plt.title("opening operation")
    # plt.tight_layout()
    # plt.show()

    # find contours in the opening image, then initialize the list of
    # contours which belong to actual characters that we will be OCR'ing
    contours = cv2.findContours(image_opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    chars = []
    # loop over the contours
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # check if contour is at least 35px wide and 100px tall, and if
        # so, consider the contour a digit
        if w >= 35 and h >= 100:
            chars.append(c)

    # compute the convex hull of the characters
    chars = np.vstack([chars[i] for i in range(0, len(chars))])
    hull = cv2.convexHull(chars)
    # allocate memory for the convex hull mask, draw the convex hull on
    # the image, and then enlarge it via a dilation
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [hull], -1, 255, -1)
    image_mask = cv2.dilate(mask, None, iterations=2)
    # plt.imshow(image_mask, cmap='gray')
    # plt.title("image mask")
    # plt.tight_layout()
    # plt.show()
    # take the bitwise of the opening image and the mask to reveal *just*
    # the characters in the image
    image_final = cv2.bitwise_and(image_opening, image_opening, mask=image_mask)

    plt.imshow(image_final, cmap='gray')
    plt.title("final image")
    plt.tight_layout()
    plt.show()

    return image_final


def tesseract_ocr(input_image):
    # OCR the input image using Tesseract
    options = "--psm 8 -c tessedit_char_whitelist=0123456789"
    image_text = pytesseract.image_to_string(input_image, config=options)
    print(image_text)

    return image_text


if __name__ == "__main__":
    params = parse_args()
    params.image_dir = os.path.join(params.image_dir, "*")

    for im in glob.glob(params.image_dir):
        filename, ext = os.path.splitext(os.path.basename(im))
        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            final_image = im_preprocess(im)
            cv2.imwrite("results/preprocess/" + filename + "_pp" + ext, final_image)
            # tesseract_ocr(final_image)
