import base64
import datetime
from datetime import datetime
import os
from math import sqrt
import cv2
import numpy as np
from flask import current_app as app


def recogniseTableFromImage(img_data, noise):
    name = convert_and_save(img_data)
    img = cv2.imread(name)
    img2 = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w < 30 and h < 30) or (w < 75) or (h<10):
            continue
        coordinates.append((x, y, w, h))
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imwrite('detecttable.jpg', img2)

    filtered = hasCloseCounture(coordinates)
    node = getMostChildren(filtered)
    (x, y, w, h) = node
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    crop_img = img[y:y + h, x:x + w]
    cv2.imwrite('croped.jpg',crop_img)

    image_bin = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    cv2.imwrite('beforeNoise.jpg', image_bin)
    if int(noise > 0):
        image_bin = noise_remove(image_bin, noise)
    cv2.imwrite('afterNoise.jpg', image_bin)

    a = 0
    b = 0
    i = len(image_bin)
    j = len(image_bin)

    while a == 0 or b == 0:
        if (w < 150) or(i <= 100 or j <= 100):
            break
        i -= 1
        j -= 1
        if image_bin[i][75] == 255 and a == 0:
            a = len(image_bin) - i
        if image_bin[j][w-75] == 255 and b == 0:
            b = len(image_bin) - j

    rotated = image_bin
    if (a != 0 or b != 0) and abs(a-b) > 20:
        x1 = abs(a-b)
        x2 = w-150
        x3 = sqrt(x1*x1+x2*x2)
        alpha = 360 - 2 * np.arctan((x1 * x1 - (x2 - x3) ** 2) / (x2 + x3) ** 2 - x1 ** 2)
        if b> a:
            alpha = 360 - alpha
        M = cv2.getRotationMatrix2D((w/2,h/2), int(alpha), 1.0)
        rotated = cv2.warpAffine(image_bin, M, (w, h))

    cv2.imwrite('rot.jpg', rotated)
    return rotated


def convert_and_save(b64_string):
    file_name = (datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) + '.png'
    directory = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    if not os.path.isdir(directory):
        os.mkdir(directory)
    with open(directory+'/'+file_name, "wb") as fh:
        fh.write(base64.decodebytes(b64_string.encode()))
    return directory+'/'+file_name


def hasCloseCounture(cordinates):
    filtered = []
    for i in cordinates:
        xi, yi, wi, hi = i
        close = 0
        for j in cordinates:
            if i == j:
                continue
            xj, yj, wj, hj = j
            d = getMinDistance([xi,xi+wi],[yi,yi+hi],[xj,xj+wj],[yj+yj+hj])
            if d<150:
                close += 1
        if close >= 2:
            filtered.append(i)
    if len(filtered) <= 5:
        return cordinates
    return filtered


def getMinDistance(xi,yi, xj,yj):
    minDistance = 999
    for x in xi:
        for y in yi:
            for x2 in xj:
                for y2 in yj:
                    d = distance(x,x2,y,y2)
                    if d<minDistance:
                        minDistance = d
    return minDistance


def distance(x1, x2, y1, y2):
    xSq = (x2-x1) ** 2
    ySq = (y2-y1) ** 2
    a = sqrt(xSq + ySq)
    return a


def sortCoordinates(coordinates):
    for i in range(0, len(coordinates)-1):
        for j in range(i+1, len(coordinates)):
            (x, y, w, h) = coordinates[i]
            (x2, y2, w2, h2) = coordinates[j]
            if w2 > w:
                temp = coordinates[i]
                coordinates[i] = coordinates[j]
                coordinates[j] = temp
    return coordinates


def getMostChildren(coordinates):
    coordinates = sortCoordinates(coordinates)
    num_of_children = {}
    for c1 in coordinates:
        children = []
        for c2 in coordinates:
            (x, y, w, h) = c1
            (x2, y2, w2, h2) = c2
            x = x-10
            y = y-10
            if x2 > x and x2 < x + w and y2 > y and y2 < y + h and w > w2:
                children.append(c2)
        num_of_children[c1] = children

    for c in coordinates:
        children = num_of_children[c]
        for child in children:
            grand_children = num_of_children[child]
            for gc in grand_children:
                if gc in children:
                    children.remove(gc)
    mostChildren = 0
    mostChildrenNode = None

    for c in coordinates:
        children = num_of_children[c]
        if len(children) > mostChildren:
            mostChildren = len(children)
            mostChildrenNode = c
    return mostChildrenNode


def noise_remove(image, noise):
    print(noise)
    copyImg = image.copy()
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            num = 0
            if image[i][j] == 255:
                if i > 0 and j > 0:
                    if image[i-1][j-1] == 255:
                        num += 1
                    if image[i-1][j] == 255:
                        num += 1
                if j+1 < len(image[i]) and i > 0:
                    if image[i-1][j+1] == 255:
                        num += 1
                if j > 0:
                    if image[i][j-1] == 255:
                        num += 1
                if j+1 < len(image[i]):
                    if image[i][j+1] == 255:
                        num += 1
                if i+1 < len(image):
                    if j > 0:
                        if image[i+1][j-1] == 255:
                            num += 1
                    if image[i+1][j] == 255:
                        num += 1
                    if j+1 < len(image[i]):
                        if image[i+1][j+1] == 255:
                            num += 1
                if num < noise:
                    copyImg[i][j] = 0
    return copyImg
