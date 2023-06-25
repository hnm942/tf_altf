import numpy as np
import os 
import sys
import json
import tensorflow as tf
import pandas as pd
from tensorflow import keras
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models.data.augment import CustomTensorOperator
from models.data.data_loader import DataLoader
from models.transformer.transformer import Transformer
from models.utils.display import DisplayOutputs

char_to_num_path = "/workspace/data/asl-fingerspelling/character_to_prediction_index.json"
data_dir = "/workspace/data/asl-fingerspelling"
predata_dir = "/workspace/data/aslfr-preprocess-dataset"

df = pd.read_csv(f'{data_dir}/train.csv')
tffiles = df.file_id.map(lambda x: f'{predata_dir}/tfds/{x}.tfrecord').unique()
dataloader = DataLoader(char_to_num_path, augment= True)

batch_size = 32


val_len = int(0.2 * len(tffiles))
# split record to train set and val set
train_dataset = tf.data.TFRecordDataset(tffiles[val_len:]).map(dataloader.load).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = tf.data.TFRecordDataset(tffiles[:val_len]).map(dataloader.load).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

batch = next(iter(train_dataset))
model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=64,
    num_layers_enc=4,
    num_layers_dec=1,
    num_classes=59,
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1,)
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss=loss_fn)

idx_to_char = list(dataloader.char_to_num.keys())
display_cb = DisplayOutputs(
    batch, dataloader.num_to_char, target_start_token_idx=2, target_end_token_idx=3
)  # set the arguments as per vocabulary index for '<' and '>'


history = model.fit(train_dataset, validation_data=val_dataset, callbacks=[display_cb], epochs=50)
