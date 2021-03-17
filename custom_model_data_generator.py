from typing import Generator

import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array

from data_generator import DataGenerator


class CustomDataGenerator(DataGenerator):
    """Data generation for image captioning model without pretrained
    feature extractor."""

    def __init__(self, path_to_annotations: str):
        super().__init__(path_to_annotations)

    def rectangle_to_square(self, input_image: np.ndarray) -> np.ndarray:
        """ Reshape image to be in squared format. """
        input_size = 126
        old_size = input_image.shape[:2]
        ratio = float(input_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        image = cv2.resize(input_image, (new_size[1], new_size[0]))
        delta_w = input_size - new_size[1]
        delta_h = input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return new_im / 255

    def __call__(self, num_photos_per_batch: int = 2) -> Generator:
        """ Data generator"""
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key, desc_list in self.descriptions.items():
                n += 1
                photo = cv2.imread(key)
                photo = self.rectangle_to_square(photo)
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
