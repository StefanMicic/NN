import collections
import json
import os

import numpy as np
import tensorflow as tf


class PreparingData:
    def __init__(self):
        self.PATH = os.path.abspath(".") + "/train2014/"
        self.annotations = self.read_annoations()
        self.image_captions_dict = self.create_image2captions()
        self.image_features_extract_model = (
            self.prepare_feature_extraction_model()
        )

    def prepare_feature_extraction_model(self):
        image_model = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet"
        )
        return tf.keras.Model(image_model.input, image_model.layers[-1].output)

    def read_annoations(self):
        with open("annotations/captions_train2014.json", "r") as f:
            annotations = json.load(f)
        return annotations

    def create_image2captions(self):
        image_captions_dict = {}

        for image_caption in self.annotations["annotations"]:
            caption = f"<START> {image_caption['caption']} <END>"
            image_path = (
                self.PATH
                + "COCO_train2014_"
                + "%012d.jpg" % (image_caption["image_id"])
            )
            try:
                image_captions_dict[image_path].append(caption)
            except Exception:
                image_captions_dict[image_path] = []
                image_captions_dict[image_path].append(caption)
        return image_captions_dict

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def create_train_set(self):
        train_captions = []
        img_name_vector = []
        train_image_paths = list(self.image_captions_dict.keys())

        for image_path in train_image_paths:
            captions = self.image_captions_dict[image_path]
            for cap in captions:
                train_captions.append(cap)
                img_name_vector.append(image_path)

        return train_captions, img_name_vector

    def load_numpy(self, img_name, cap):
        img_tensor = np.load(img_name.decode("utf-8") + ".npy")
        return img_tensor, cap

    def tokenize_data(self, train_captions):
        top_k = 5000
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=top_k,
            oov_token="<unk>",
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
        )
        tokenizer.fit_on_texts(train_captions)

        train_seqs = tokenizer.texts_to_sequences(train_captions)
        return train_seqs

    def create_final_dataset(
        self, img_name_train, cap_train, BATCH_SIZE=64, BUFFER_SIZE=1000
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (img_name_train, cap_train)
        )

        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                self.load_numpy, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def __call__(self, BATCH_SIZE=64, BUFFER_SIZE=1000):
        train_captions, img_name_vector = self.create_train_set()

        encode_train = sorted(set(img_name_vector))

        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            self.load_image, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(16)

        for img, path in image_dataset:
            batch_features = self.image_features_extract_model(img)
            batch_features = tf.reshape(
                batch_features,
                (batch_features.shape[0], -1, batch_features.shape[3]),
            )

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

        train_seqs = self.tokenize_data(train_captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding="post"
        )

        img_to_cap_vector = {}
        for img, cap in zip(img_name_vector, cap_vector):
            try:
                img_to_cap_vector[img].append(cap)
            except Exception:
                img_to_cap_vector[img] = []
                img_to_cap_vector[img].append(cap)

        img_name_train = []
        cap_train = []
        for imgt in list(img_to_cap_vector.keys()):
            capt_len = len(img_to_cap_vector[imgt])
            img_name_train.extend([imgt] * capt_len)
            cap_train.extend(img_to_cap_vector[imgt])

        return self.create_final_dataset(
            img_name_train, cap_train, BATCH_SIZE, BUFFER_SIZE
        )
