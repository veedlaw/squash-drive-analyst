import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import deflicker


def get_blocks3D(arr, num_row_blocks=4, num_col_blocks=5):
    """
    :param arr: 3D Array to partition into blocks
    :param num_row_blocks: Number of blocks to generate from a row
    :param num_col_blocks: Number of blocks to generate from a column
    :return: Generates num_row_blocks * num_col_blocks equally-sized pieces of the array
    """

    height = arr.shape[0]
    width = arr.shape[1]

    stride_row = int(width / num_row_blocks)
    stride_col = int(height / num_col_blocks)

    for x in range(0, arr.shape[0], stride_col):
        for y in range(0, arr.shape[1], stride_row):
            yield arr[x:x + stride_col, y:y + stride_row, :]


def get_blocks2D(arr, num_row_blocks=4, num_col_blocks=5):
    """
    :param arr: Array to partition into blocks
    :param num_row_blocks: Number of blocks to generate from a row
    :param num_col_blocks: Number of blocks to generate from a column
    :return: Generates num_row_blocks * num_col_blocks equally-sized pieces of the array
    """

    height = arr.shape[0]
    width = arr.shape[1]

    stride_row = int(width / num_row_blocks)
    stride_col = int(height / num_col_blocks)

    for x in range(0, arr.shape[0], stride_col):
        for y in range(0, arr.shape[1], stride_row):
            yield arr[x:x + stride_col, y:y + stride_row]


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=4, pxstep=90, pystep=128):
    """
    Draws a grid on an image
    :param img: Image for the lines to be drawn on.
    :param line_color: Color of the line
    :param thickness: Thickness of the line
    :param type_: Type of line
    :param pxstep: Every pxstep pixels a line will be drawn
    :param pystep: Every pystep pixels a line will be drawn
    """
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep
    return img


def show_histogram(img, processed_img):
    """Debugging function that shows plotted histogram of two images."""
    hist_full = cv.calcHist([img], [0], None, [255], [0, 255])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(processed_img, 'gray')
    plt.subplot(223), plt.plot(hist_full)
    plt.xlim([0, 255])
    plt.show()


def get_deflicker_parameters(video_reader):
    # region initialize
    initial_frame = next(video_reader.get_frame())
    if initial_frame is None:
        return

    frame_width = initial_frame.shape[0]
    frame_height = initial_frame.shape[1]
    num_frames_to_read = 30
    # endregion initialize

    deflickerer = deflicker.Deflicker(frame_width, frame_height, num_frames_to_read)
    deflickerer.append_pixel_intensity_data(initial_frame)

    for i in range(num_frames_to_read):
        frame = next(video_reader.get_frame())

        if frame is None:
            return

        deflicker.append_pixel_intensity_data(frame)
    block_thresholds = deflicker.choose_pixels_to_follow()
    video_reader.set_stream_frame_pos(0)

    return block_thresholds


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


def draw_contours(frame, preprocessed, detector):
    new_rectangles = detector.classify(preprocessed)

    # max_area = sys.maxsize
    min_area = sys.maxsize
    for rect in new_rectangles:
        x, y, w, h = rect
        area = w * h
        if area < min_area:
            min_area = area

    for rect in new_rectangles:
        x, y, w, h = rect
        if w * h == min_area:
            img = cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 3)
        else:
            img = cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    # return blend(images)
    return frame


def rect_area(rect):
    x, y, width, height = rect
    return x * width + y * height

def rect_center(rect):
    x, y, width, height = rect
    return None