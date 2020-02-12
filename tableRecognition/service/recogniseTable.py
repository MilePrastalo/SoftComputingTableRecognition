import base64
import datetime
from datetime import datetime
import os
from math import sqrt

import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import current_app as app


def recogniseTableFromImage(img_data):
    name = convert_and_save(img_data)
    img = cv2.imread(name)
    img2 = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, image_bin = cv2.threshold(img,110, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    cv2.imwrite('bin.jpg', image_bin)
    contoures, _ = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for cnt in contoures:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 and h < 30:
            continue
        if w < 50:
            continue
        coordinates.append((x, y, w, h))
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite('detecttable.jpg', img2)

    filtered = hasCloseCounture(coordinates)
    node = getMostChildren(filtered)
    (x,y,w,h) = node
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    crop_img = img[y:y + h, x:x + w]
    cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
    cv2.imwrite('croped.jpg',crop_img)
    return img


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
