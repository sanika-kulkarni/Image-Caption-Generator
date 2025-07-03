import string
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import os

def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in os.listdir(directory):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[name] = feature
    return features

def load_descriptions(filepath):
    mapping = {}
    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            image_id, caption = tokens[0].split('#')[0], tokens[1]
            caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
            mapping.setdefault(image_id, []).append('startseq ' + caption + ' endseq')
    return mapping

def to_lines(descriptions):
    return [desc for key in descriptions for desc in descriptions[key]]

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)
