import numpy as np
import os 
import sys
import json
import tensorflow as tf
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models.data.augment import CustomTensorOperator, RandomInterpolation
from models.data.data_loader import DataLoader
char_to_num_path = "/workspace/data/asl-fingerspelling/character_to_prediction_index.json"
data_dir = "/workspace/data/asl-fingerspelling"
predata_dir = "/workspace/data/aslfr-preprocess-dataset"

df = pd.read_csv(f'{data_dir}/train.csv')
tffiles = df.file_id.map(lambda x: f'{predata_dir}/tfds/{x}.tfrecord').unique()

dataset = tf.data.TFRecordDataset(tffiles[1])
sub_dataset =dataset.take(1)
dataloader = DataLoader(char_to_num_path)

for record in sub_dataset:
    landmarks, pharse= dataloader.load(record.numpy())

print("################################################# test augmentation ##################################################")
batch_size = tf.shape(landmarks)[0]
num_landmarks = tf.shape(landmarks)[1]
max_translation = 10
max_rotation = 10
max_scale = 0.1
translation = tf.random.uniform((batch_size, 2), minval=-max_translation, maxval=max_translation)
rotation = tf.random.uniform((batch_size,), minval=-max_rotation, maxval=max_rotation)
scale = tf.random.uniform((batch_size,), minval=1 - max_scale, maxval=1 + max_scale)
scale_range = (0.8, 1.2)
shift_range = (-1, 1)
random_interpolation =  RandomInterpolation.RandomInterpolation(scale_range, shift_range)
transformed_landmarks = []
for i in range(batch_size):
    current_landmarks = tf.reshape(landmarks[i], [-1, 3])
    # print(current_landmarks.shape)
    # Translation
    translation_operator = CustomTensorOperator.Translation(translation[i])
    translated_landmarks = translation_operator.translator(current_landmarks)

    # Rotation
    rotation_operator = CustomTensorOperator.Rotation(rotation[i])
    rotated_landmarks = rotation_operator.rotation(translated_landmarks)

    # Scale
    scale_operator = CustomTensorOperator.Scale(scale[i])
    scaled_landmarks = scale_operator.scale(rotated_landmarks)
    flatten_landmarks = tf.reshape(scaled_landmarks, [342])
    transformed_landmarks.append(scaled_landmarks)

transformed_landmarks = tf.stack(transformed_landmarks)
