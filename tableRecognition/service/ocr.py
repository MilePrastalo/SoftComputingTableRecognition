import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
from service import levenshtein
# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
# KMeans
from sklearn.cluster import KMeans

# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16, 12

alphabet = "QWERTYUIOPASDFGHJKLZXCVBNMŠĐČĆŽqwertyuiopasdfghjklzxcvbnmšđčćž%0123456789\,/<>"
alphabet = [c for c in alphabet]

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.int32)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train, epochs):
    x_train = np.array(x_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def get_hooks(heights):
    hooks = []
    for idx, h in enumerate(heights):
        if 6 < h < 13:
            hooks.append(idx)
    return hooks


def sort_regions(regions):
    regions = sorted(regions, key=lambda x: x[1][1])
    region_heights = [region[1][3] for region in regions]
    region_heights = np.array(region_heights).reshape(len(region_heights), 1)
    hooks = get_hooks(region_heights)
    breaks = [0]
    for i in range(1, len(regions) - 1):
        current_reg = regions[i]
        next_reg = regions[i + 1]
        # nextY > currY +curH
        if (next_reg[1][1] > (current_reg[1][1] + current_reg[1][3])) and (i not in hooks):
            breaks.append(i + 1)
        elif (next_reg[1][1] > (current_reg[1][1] + current_reg[1][3]) and (i in hooks)
              and (current_reg[1][1] - current_reg[1][3]) >= regions[i - 1][1][1]):
            breaks.append(i + 1)
    breaks.append(len(regions))
    print(breaks)
    for i in range(len(breaks) - 1):
        regions[breaks[i]:breaks[i + 1]] = sorted(regions[breaks[i]:breaks[i + 1]], key=lambda x: x[1][0])
    return regions


def select_roi_with_distances(image_orig, image_bin, help_):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < help_ and 5 < h < image_orig.shape[0]:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h + 1)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sort_regions(regions_array)
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # x_next - (x_current + w_current)
        region_distances.append(distance)
    return regions_array, region_distances


def get_spaces(distances):
    spaces = []
    for idx, space in enumerate(distances):
        if 17 < space < 27:
            spaces.append(idx)
    return spaces


def get_newline(distances):
    newlines = []
    for idx, space in enumerate(distances):
        if space < 0 and abs(space) > 60:
            newlines.append(idx)
    return newlines

def display_result_with_spaces(outputs, alphabet, distances):
    result = alphabet[winner(outputs[0])]
    spaces = get_spaces(distances)
    newlines = get_newline(distances)
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if idx in spaces:
            result += ' '
        elif idx in newlines:
            result += '\n'
        result += alphabet[winner(output)]
    return result


def help_function(idx, last_idx, regions):
    left = -1
    if idx == 0:
        left = 1
    elif idx == last_idx and idx != 0:
        left = -1
    else:
        razlika_levo = abs(regions[idx][1][0] - regions[idx - 1][1][0])
        razlika_desno = abs(regions[idx][1][0] - regions[idx + 1][1][0])
        space = (regions[idx][1][0] + regions[idx][1][2]) < regions[idx + 1][1][0]
        if razlika_levo <= razlika_desno:
            left = -1
        elif (razlika_levo > razlika_desno and ((regions[idx][1][0] + regions[idx][1][2]) <= regions[idx + 1][1][0])):
            left = -1
        else:
            left = 1
    return left


def detect_serbian_letters(sorted_regions, img_bin):
    region_heights = [region[1][3] for region in sorted_regions]
    region_heights = np.array(region_heights).reshape(len(region_heights), 1)

    hooks = get_hooks(region_heights)
    index_to_avoid = []
    for i, region in enumerate(sorted_regions):
        flag = help_function(i, len(sorted_regions) - 1, sorted_regions)
        if ((i in hooks)
                and ((sorted_regions[i][1][1] + sorted_regions[i][1][3]) <= sorted_regions[i + flag][1][1])):
            y = sorted_regions[i][1][1]
            x = sorted_regions[i + flag][1][0]
            w = sorted_regions[i + flag][1][2]
            if (x + w) < (sorted_regions[i][1][0] + sorted_regions[i][1][2]):
                w = w + ((sorted_regions[i][1][0] + sorted_regions[i][1][2]) - (x + w))
            h = sorted_regions[i + flag][1][3] + sorted_regions[i][1][3] + 5
            region = img_bin[y:y + h + 1, x:x + w + 1]
            sorted_regions[i + flag] = [resize_region(region), (x, y, w, h)]
            # remove
            index_to_avoid.append(i)
        elif ((i in hooks) and (
                (sorted_regions[i][1][1] + sorted_regions[i][1][3]) >= sorted_regions[i - 1][1][1])):
            # remove
            index_to_avoid.append(i)
        # procenat
        elif ((i not in hooks) and
              (sorted_regions[i][1][3] > 18) and (sorted_regions[i][1][3] < 28)):
            y = sorted_regions[i + flag][1][1]
            x = sorted_regions[i][1][0]
            w = sorted_regions[i + flag][1][2] + (sorted_regions[i + flag][1][0] - sorted_regions[i][1][0])
            if flag == -1:
                x = sorted_regions[i + flag][1][0]
                w = sorted_regions[i + flag][1][2] + (sorted_regions[i][1][0] + sorted_regions[i][1][2]) - (
                        sorted_regions[i + flag][1][0] + sorted_regions[i + flag][1][2])
            h = sorted_regions[i + flag][1][3]
            region = img_bin[y:y + h + 1, x:x + w + 1]
            sorted_regions[i + flag] = [resize_region(region), (x, y, w, h)]
            index_to_avoid.append(i)
    sorted_regions = [reg for idx, reg in enumerate(sorted_regions) if idx not in index_to_avoid]
    return sorted_regions


def calculate_distance(regions_array):
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # x_next - (x_current + w_current)
        region_distances.append(distance)
    return region_distances


def draw_regions(img, regions, img_name):
    for region in regions:
        cv2.rectangle(img, (region[1][0], region[1][1]), (region[1][0] + region[1][2], region[1][1] + region[1][3]),
                      (0, 255, 0), 2)
    cv2.imwrite("C:/Users/Jelena Cuk/Desktop/SOFT/SoftComputingTableRecognition/tableRecognition/ocr_results/" + img_name, img)


def train_nn(alphabet, regions, ann):
    outputs = convert_output(alphabet)
    regions = [region[0] for region in regions]
    inputs = prepare_for_ann(regions)
    ann = train_ann(ann, inputs, outputs, 8000)
    return ann


def ocr(ann, regions, alphabet, distances):
    regions = [region[0] for region in regions]
    inputs = prepare_for_ann(regions)
    result = ann.predict(np.array(inputs, np.float32))
    res = display_result_with_spaces(result, alphabet, distances)
    return res


def training():
    print('ALPHABET SIZE = ' + str(len(alphabet)))
    filename = "C:/Users/Jelena Cuk/Desktop/SOFT PROJEKAT/SoftComputingTableRecognition/tableRecognition/ocr_results/train_arial.png"
    img_org = load_image(filename)
    img_gray = image_gray(img_org)
    img_bin = image_bin(img_gray)
    img_bin = invert(img_bin)

    regions, distances = select_roi_with_distances(img_org.copy(), img_bin, img_org.shape[1])
    print('REGIONS SIZE BEFORE SERBIAN = ' + str(len(regions)))
    draw_regions(img_org.copy(), regions, 'before_serbian.png')
    regions = detect_serbian_letters(regions.copy(), img_bin)
    print('REGIONS SIZE AFTER SERBIAN = ' + str(len(regions)))
    draw_regions(img_org, regions, 'after_serbian.png')
    ann = create_ann(len(alphabet))
    ann = train_nn(alphabet, regions, ann)
    model_json = ann.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ann.save_weights("model.h5")
    print("Saved model to disk")
    return ann


def load_model_from_disk():
    json_file = open('service/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ann = model_from_json(loaded_model_json)
    # load weights into new model
    ann.load_weights("service/model.h5")
    print("Loaded model from disk")
    return ann

def get_text(filename):
    img_org = load_image(filename)
    img_gray = image_gray(img_org)
    img_bin = image_bin(img_gray)
    #img_bin = invert(img_bin)
    #ret, img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    regions, distances = select_roi_with_distances(img_org.copy(), img_bin, 100)
    #regions = detect_serbian_letters(regions, img_bin)
    print('REGIONS = ' + str(len(regions)))
    draw_regions(img_org, regions, 'VEZBA' + '.png')
    distances = calculate_distance(regions)
    print('DISTANCES = ' + str(distances))
    ann = load_model_from_disk()
    res = ocr(ann, regions, alphabet, distances)
    return res

def table_ocr(table):

    # ann = training(alphabet)
    ann = load_model_from_disk()
    table_text = ""
    matrica = []
    for idx1, row in enumerate(table):
        row_text = ''
        matrica.append([])
        for i, column in enumerate(row):
            res = 'xxx'
            try:
                # img_gray = image_gray(column)
                ret, img_bin = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # regions
                regions, distances = select_roi_with_distances(column.copy(), img_bin, 60)
                regions = detect_serbian_letters(regions, img_bin)
                draw_regions(column, regions, 'regions_' + str(idx1+i) + '.png')
                # calculate distances
                distances = calculate_distance(regions)
                res = ocr(ann, regions, alphabet, distances)
                print('PREPOZNAO => ' + res)
                res = levenshtein.convert_to_real_word(res)
                print('LEVENSHTEIN => ' + res)
            except Exception:
                pass
            matrica[idx1].append(res)
            row_text = row_text + " " + res
        table_text = table_text + row_text + '\n'
    return table_text, matrica


#if __name__ == '__main__':

    #res = get_text("C:/Users/Jelena Cuk/Desktop/TEST SKUP/DVA_reda_BROJEVI.png")
    #print(res)
    #training()
# ann = load_model_from_disk()
# text = get_text(filename, ann, alphabet)
# print(text)
