import collections
import json

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image


class DataGenerator:
    def __init__(self, PATH: str):
        model = InceptionV3(weights="imagenet")
        self.model_new = Model(model.input, model.layers[-2].output)
        self.descriptions, self.train = self.read_file(PATH)
        self.all_train_captions = self.prepare_for_training()
        self.vocab = self.create_vocab()

    def read_file(self, PATH: str):
        with open("annotations/captions_val2014.json", "r") as f:
            annotations = json.load(f)

        descriptions = collections.defaultdict(list)
        image_paths = set()
        for val in annotations["annotations"]:
            caption = f"<start> {val['caption']} <end>"
            image_path = (
                PATH + "COCO_val2014_" + "%012d.jpg" % (val["image_id"])
            )
            descriptions[image_path].append(caption)
            image_paths.add(image_path)

        train_image_paths = list(image_paths)[:20]
        train = []
        for image_path in train_image_paths:
            caption_list = descriptions[image_path]
            train.extend([image_path] * len(caption_list))
        return descriptions, train

    def prepare_for_training(self):
        train_descriptions = dict()
        all_train_captions = []

        for key, desc_list in self.descriptions.items():
            for desc in desc_list:
                if key in self.train:
                    if key not in train_descriptions:
                        train_descriptions[key] = list()
                    desc = "".join(desc)
                    train_descriptions[key].append(desc)
                    all_train_captions.append(desc)
        return all_train_captions

    def create_vocab(self):
        word_count_threshold = 5
        word_counts = {}
        vocab = set()
        for sent in self.all_train_captions:
            for w in sent.split(" "):
                word_counts[w] = word_counts.get(w, 0) + 1
                if word_counts[w] >= word_count_threshold:
                    vocab.add(w)
        return vocab

    def create_matrix(self):
        ixtoword = {}
        wordtoix = {}
        ix = 1
        for w in self.vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        vocab_size = len(ixtoword) + 1
        return vocab_size

    def calculate_max_length(self):
        max_length = max(len(d.split()) for d in self.all_train_captions)
        return max_length

    def encode(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fea_vec = self.model_new.predict(x)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec

    def extract_features(self):
        encoding_train = {}
        for img in self.train:
            encoding_train[img] = self.encode(img)

        return encoding_train
