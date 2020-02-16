import numpy as np
import cv2  # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections

# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16, 12

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
# KMeans
from sklearn.cluster import KMeans


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


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def sort_regions(regions):
    regions = sorted(regions, key=lambda x: x[1][1])
    labels = np.zeros(len(regions))
    kukice_group = 1;
    try:
        region_heights = [region[1][3] for region in regions]
        region_heights = np.array(region_heights).reshape(len(region_heights), 1)
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(region_heights)
        kukice_group = min(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
        if (abs(k_means.cluster_centers_[0] - k_means.cluster_centers_[1]) > 14):
            labels = k_means.labels_

    except Exception:
        pass
    breaks = [0]
    for i in range(1, len(regions) - 1):
        current_reg = regions[i]
        next_reg = regions[i + 1]
        # nextY > currY +curH
        if ((next_reg[1][1] > (current_reg[1][1] + current_reg[1][3])) and (labels[i] != kukice_group)):
            breaks.append(i + 1)
        elif (next_reg[1][1] > (current_reg[1][1] + current_reg[1][3]) and (labels[i] == kukice_group)
              and (current_reg[1][1] - current_reg[1][3]) >= regions[i - 1][1][1]):
            breaks.append(i + 1)
    breaks.append(len(regions))
    for i in range(len(breaks) - 1):
        regions[breaks[i]:breaks[i + 1]] = sorted(regions[breaks[i]:breaks[i + 1]], key=lambda x: x[1][0])
    return regions


def select_roi_with_distances(image_orig, image_bin, help_):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < help_ and h > 7):
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h + 1)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sort_regions(regions_array)

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

    return regions_array, region_distances


def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def help_function(idx, last_idx, regions):
    left = -1
    if idx == 0:
        left = 1
    elif idx == last_idx and idx != 0:
        left = -1
    else:
        diff_left = abs(regions[idx][1][0] - regions[idx - 1][1][0])
        diff_right = abs(regions[idx][1][0] - regions[idx + 1][1][0])
        if diff_left <= diff_right:
            left = -1
        elif diff_left > diff_right and ((regions[idx][1][0] + regions[idx][1][2]) <= regions[idx + 1][1][0]):
            left = -1
        else:
            left = 1
    return left


def detect_serbian_letters(sorted_regions, img_bin):

    #Izdvajamo samo visine i primenjujemo K-means
    # tri klastera: velika  slova, mala slova i kukice
    heights = [region[1][3] for region in sorted_regions]
    heights = np.array(heights).reshape(len(heights), 1)
    k_means = KMeans(n_clusters=3, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(heights)
    hooks = min(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]

    #Prolazimo kroz izdvojene regione i ako je region kukica
    #spajamo ga sa prethodnim/sledecim regionom
    remove = []
    for i, region in enumerate(sorted_regions):
        flag = help_function(i, len(sorted_regions) - 1, sorted_regions)
        if ((k_means.labels_[i] == hooks)
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
            remove.append(i)
        elif ((k_means.labels_[i] == hooks) and (
                (sorted_regions[i][1][1] + sorted_regions[i][1][3]) >= sorted_regions[i - 1][1][1])):
            remove.append(i)
    sorted_regions = [reg for idx, reg in enumerate(sorted_regions) if idx not in remove]
    return sorted_regions


def calculate_distance(regions_array):

    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        # x_next - (x_current + w_current)
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    return region_distances


def draw_regions(img, regions):
    for region in regions:
        cv2.rectangle(img, (region[1][0], region[1][1]), (region[1][0] + region[1][2], region[1][1] + region[1][3]),
                      (0, 255, 0), 2)
    display_image(img)


def train_nn(alphabet, regions, ann):
    outputs = convert_output(alphabet)
    regions = [region[0] for region in regions]
    inputs = prepare_for_ann(regions)
    ann = train_ann(ann, inputs, outputs, 4000)
    return ann


def ocr(ann, regions, alphabet, distances):
    regions = [region[0] for region in regions]
    inputs = prepare_for_ann(regions)
    result = ann.predict(np.array(inputs, np.float32))
    k_means = KMeans(n_clusters=3, max_iter=2000, tol=0.00001, n_init=10)
    distances = np.array(distances).reshape(len(distances), 1)
    k_means.fit(distances)
    res = display_result_with_spaces(result, alphabet, k_means)
    return res

def training(alphabet):

    print('ALPHABET SIZE = ' + str(len(alphabet)))
    filename = "C:/Users/Jelena Cuk/Desktop/SOFT/SoftComputingTableRecognition/tableRecognition/images/train_arial_space.png"
    img_org = load_image(filename)
    img_gray = image_gray(img_org)
    img_bin = image_bin(img_gray)
    img_bin = invert(img_bin)

    regions, distances = select_roi_with_distances(img_org.copy(), img_bin, img_org.shape[1])
    print('REGIONS SIZE BEFORE SERBIAN = ' + str(len(regions)))
    regions = detect_serbian_letters(regions, img_bin)
    print('REGIONS SIZE AFTER SERBIAN = ' + str(len(regions)))
    #draw_regions(img_org, regions)
    ann = create_ann(len(alphabet))
    ann = train_nn(alphabet, regions, ann)
    model_json = ann.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ann.save_weights("model.h5")
    print("Saved model to disk")
    return ann

def get_text(filename, ann, alphabet):
    img_org = load_image(filename)
    img_gray = image_gray(img_org)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    regions, distances = select_roi_with_distances(img_org.copy(), img_bin, 60)
    regions = detect_serbian_letters(regions, img_bin)
    #draw_regions(img_org, regions)
    distances = calculate_distance(regions)
    res = ocr(ann, regions, alphabet, distances)
    return res

def load_model_from_disk():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ann = model_from_json(loaded_model_json)
    # load weights into new model
    ann.load_weights("model.h5")
    print("Loaded model from disk")
    return ann

def table_ocr(table):
    alphabet = "QWERTYUIOPASDFGHJKLZXCVBNMŠĐČĆŽqwertyuiopasdfghjklzxcvbnmšđčćž0123456789"
    alphabet = [c for c in alphabet]
    #ann = training(alphabet)
    ann = load_model_from_disk()
    table_text = ""
    for row in table:
        row_text = ''
        for i, column in enumerate(row):
            #cv2.imwrite("cropped/" + str(i) + '.png', column)
            #img_gray = image_gray(column)
            ret, img_bin = cv2.threshold(column, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            regions, distances = select_roi_with_distances(column.copy(), img_bin, 60)
            regions = detect_serbian_letters(regions, img_bin)
            distances = calculate_distance(regions)
            res = ocr(ann, regions, alphabet, distances)
            row_text = row_text + " " + res
        table_text = table_text + row_text + '\n'
    return table_text



#if __name__ == '__main__':


    #ann = training(alphabet)
    #ann = load_model_from_disk()
    #filename = 'C:/Users/Jelena Cuk/Desktop/SOFT/SoftComputingTableRecognition/tableRecognition/images/3.png'
    #text = get_text(filename, ann, alphabet)
    #print(text)

