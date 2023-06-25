import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

from models.data.augment import CustomTensorOperator, RandomInterpolation

class DataLoader:
    def __init__(self, char_to_num_path, max_translation = 10, max_rotation = 10, max_scale = 0.1, augment = False, LIP = None, frame_len = 128, batch_size = 32):
        if LIP is None:
            self.LIP = [
                    61, 185, 40, 39, 37, 267, 269, 270, 409,
                    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                ]
        else:
            self.LIP = LIP
        self.frame_len = frame_len
        self.batch_size = batch_size 
        self.augment = augment
        self.load_character_to_prediction_index(char_to_num_path)
        self.load_landmark_interesting_points()
        self.initStaticHashTable()
        self.max_translation =  max_translation
        self.max_rotation = max_rotation
        self.max_scale = max_scale

    def load_landmark_interesting_points(self):
        self.FACE = [f'x_face_{i}' for i in self.LIP] + [f'y_face_{i}' for i in self.LIP] + [f'z_face_{i}' for i in self.LIP]
        self.LHAND = [f'x_left_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)]
        self.RHAND = [f'x_right_hand_{i}' for i in range(21)] + [f'y_right_hand_{i}' for i in range(21)] + [f'z_right_hand_{i}' for i in range(21)]
        self.POSE = [f'x_pose_{i}' for i in range(33)] + [f'y_pose_{i}' for i in range(33)] + [f'z_pose_{i}' for i in range(33)]
        self.X = [f'x_face_{i}' for i in self.LIP] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_right_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in range(33)]
        self.Y = [f'y_face_{i}' for i in self.LIP] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_right_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in range(33)]
        self.Z = [f'z_face_{i}' for i in self.LIP] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_right_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in range(33)]
        self.sel_col = self.X + self.Y + self.Z

    def load_character_to_prediction_index(self,char_to_num_path):
        with open (char_to_num_path, "r") as f:
            self.char_to_num = json.load(f)
            self.num_to_char = {j:i for i,j in self.char_to_num.items()}
        
    def initStaticHashTable(self):
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self.char_to_num.keys()),
                values=list(self.char_to_num.values()),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )
        self.table.export()        

    def load(self, record_bytes):
    #     print(record_bytes)
        schema = {COL: tf.io.FixedLenFeature([self.frame_len], dtype=tf.float32) for COL in self.sel_col}
        schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
        features = tf.io.parse_single_example(record_bytes, schema)
        phrase = features["phrase"]
        landmarks = tf.convert_to_tensor([features[COL] for COL in self.sel_col])
        landmarks = tf.transpose(landmarks)
        if self.augment == True:
            batch_size = tf.shape(landmarks)[0]
            num_landmarks = tf.shape(landmarks)[1]
            translation = tf.random.uniform((batch_size, 2), minval= -self.max_translation, maxval=self.max_translation)
            rotation = tf.random.uniform((batch_size,), minval= -self.max_rotation, maxval= self.max_rotation)
            scale = tf.random.uniform((batch_size,), minval=1 - self.max_scale, maxval=1 + self.max_scale)
            scale_range = (0.8, 1.2)
            shift_range = (-2, 2)
            random_interpolator = RandomInterpolation.RandomInterpolation(scale_range, shift_range)
            transformed_landmarks = []
            for i in range(batch_size):
                current_landmarks = tf.reshape(landmarks[i], [-1, 3])
                # print(current_landmarks.shape)
                # interpolation
                interpolator_landmarks = random_interpolator.scale_and_shift(current_landmarks)
                # Translation
                translation_operator = CustomTensorOperator.Translation(translation[i])
                translated_landmarks = translation_operator.translator(interpolator_landmarks)

                # Rotation
                rotation_operator = CustomTensorOperator.Rotation(rotation[i])
                rotated_landmarks = rotation_operator.rotation(translated_landmarks)

                # Scale
                scale_operator = CustomTensorOperator.Scale(scale[i])
                scaled_landmarks = scale_operator.scale(rotated_landmarks)
                flatten_landmarks = tf.reshape(scaled_landmarks, [342])
                transformed_landmarks.append(scaled_landmarks)

        mask = tf.math.less(landmarks, -2)
    #     nan_tensor = tf.fill(tf.shape(landmarks), tf.constant(np.nan, dtype=tf.float32))
        nan_tensor = tf.fill(tf.shape(landmarks), tf.constant(0, dtype=tf.float32))
        landmarks = tf.where(mask, nan_tensor, landmarks)
        
        phrase = '#' + phrase + '$'
        phrase = tf.strings.bytes_split(phrase)
        phrase = self.table.lookup(phrase)
        phrase = tf.pad(phrase, paddings=[[0, 64 - tf.shape(phrase)[0]]])
        return landmarks, phrase



