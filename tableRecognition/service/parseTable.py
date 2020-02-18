import cv2.cv2 as cv2
import numpy as np


def parse_table(table_img):
    cv2.imwrite("table_img.jpg", table_img)

    # Horizontal kernel of (kernel_length X 1), which will detect all the horizontal lines from the image.
    horizontal_size = np.array(table_img).shape[0] // 30
    print("H size: ", horizontal_size)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Vertical kernel of (1 X kernel_length), which will detect all the vertical lines from the image.
    vertical_size = np.array(table_img).shape[1] // 30
    print("V size: ", vertical_size)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

    # A kernel of (7 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(table_img, vertical_structure, iterations=3)
    cv2.imwrite("img_templ.jpg", img_temp1)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_structure, iterations=3)
    cv2.imwrite("verticle_lines.jpg", vertical_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(table_img, horizontal_structure, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_structure, iterations=6)
    cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter
    # to get a third image as summation of two images.
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    cv2.imwrite("img_final_bin1.jpg", img_final_bin)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=4)
    cv2.imwrite("img_final_bin2.jpg", img_final_bin)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 254, 255, cv2.THRESH_BINARY)
    cv2.imwrite("img_final_bin3.jpg", img_final_bin)

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    # Calculates average height of contours
    h_cnt = 0
    average_h = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        average_h += h
        h_cnt += 1

    if h_cnt > 0:
        average_h = average_h / h_cnt
    else:
        return

    print("average h: ", average_h)
    valid_contours = []
    for c in contours:

        # Returns the location and width,height for every contour
        # If the box height is greater then 50 or height > average height - 7,
        x, y, w, h = cv2.boundingRect(c)

        if w > 200 and (h > 50 or h > average_h - 7):
            valid_contours.append(c)

    # Recreates table positions with cropped contours of original table.
    cropped_matrix = recreate_table(valid_contours)

    matrix_copy = []

    print("MATRIX")
    for idx1, niz in enumerate(cropped_matrix):
        print('=== RED ===')
        matrix_copy.append([])
        for idx2, item in enumerate(niz):
            x, y, w, h = cv2.boundingRect(item)
            print(x, y, w, h)
            # Adds cropped images to positions of copied recreated table.
            matrix_copy[idx1].append(table_img[y:y + h - 7, x:x + w])
            cv2.imwrite("cropped/" + str(idx1) + '___' + str(idx2) + '.png', table_img[y:y + h - 7, x:x + w])

    return matrix_copy


def recreate_table(contours):
    breaks = list()
    # Adds first table element.
    breaks.append(0)

    # Decides in which row should the element be.
    for i in range(0, len(contours) - 1):
        x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(contours[i])
        x_next, y_next, w_next, h_next = cv2.boundingRect(contours[i + 1])

        if (y_curr - 15) < y_next < (y_curr + 15):
            continue
        else:
            breaks.append(i + 1)

    # Adds last table element.
    breaks.append(len(contours))

    # Creates the table matrix and feels it with contour indexes.
    # Also sorts it from left to right.
    matrix = []
    for i in range(len(breaks) - 1):
        print(i, breaks[i])
        if breaks[i] + 1 == len(contours):
            matrix.append([contours[breaks[i]]])
            break
        if i == len(breaks) - 1:
            break
        x_curr = cv2.boundingRect(contours[breaks[i]])
        x_next = cv2.boundingRect(contours[breaks[i] + 1])

        # Checks which one is first, left or right.
        if x_curr > x_next:
            matrix.append(reversed(contours[breaks[i]:breaks[i + 1]]))
        else:
            matrix.append((contours[breaks[i]:breaks[i + 1]]))

    return matrix


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
