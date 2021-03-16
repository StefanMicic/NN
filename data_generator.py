import json

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array


class DataGenerator:
    def __init__(self, path_to_annotations: str):
        with open(path_to_annotations, "r") as f:
            self.descriptions = json.load(f)
        self.image_paths, self.train_captions = self.get_image_paths()
        self.max_length = max(len(d.split()) for d in self.train_captions)
        self.vocab = self.get_vocabulary()
        self.ixtoword, self.wordtoix = self.create_tokenization_matrix()
        self.model_new = self.create_encoder()
        self.encoding_train = self.get_encoded_features()
        self.vocab_size = len(self.ixtoword) + 1

    def __call__(self, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key, desc_list in self.descriptions.items():
                n += 1
                photo = self.encoding_train[key]
                for desc in desc_list:
                    seq = [
                        self.wordtoix[word]
                        for word in desc.split(" ")
                        if word in self.wordtoix
                    ]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences(
                            [in_seq], maxlen=self.max_length
                        )[0]
                        out_seq = to_categorical(
                            [out_seq], num_classes=self.vocab_size
                        )[0]
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)

                if n == num_photos_per_batch:
                    yield ([array(X1), array(X2)], array(y))
                    X1, X2, y = list(), list(), list()
                    n = 0

    def get_encoded_features(self):
        encoding_train = {}
        for img in self.image_paths:
            encoding_train[img] = self.encode(img)
        return encoding_train

    def encode(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fea_vec = self.model_new.predict(x)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

    def create_encoder(self):
        model = InceptionV3(weights="imagenet")
        return Model(model.input, model.layers[-2].output)

    def get_image_paths(self):
        image_paths = set()
        for image_path in self.descriptions.keys():
            image_paths.add(image_path)

        train_captions = []
        for image_path in list(image_paths):
            caption_list = self.descriptions[image_path]
            train_captions.extend(caption_list)

        return image_paths, train_captions

    def get_vocabulary(self):
        word_count_threshold = 5

        word_counts = {}
        vocab = set()
        for sent in self.train_captions:
            for w in sent.split(" "):
                word_counts[w] = word_counts.get(w, 0) + 1
                if word_counts[w] >= word_count_threshold:
                    vocab.add(w)
        return vocab

    def create_tokenization_matrix(self):
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in self.vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        return ixtoword, wordtoix
