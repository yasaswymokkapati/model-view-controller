import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 2500, train_size = 7500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    im_bw = im_pil.convert('L')
    im_bw_resized = im_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(im_bw_resized, pixel_filter)
    im_bw_resized_inverted_scaled = np.clip(im_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(im_bw_resized)
    im_bw_resized_inverted_scaled = np.asarray(im_bw_resized_inverted_scaled) / max_pixel
    test_sample = np.array(im_bw_resized_inverted_scaled).reshape(1, 784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]