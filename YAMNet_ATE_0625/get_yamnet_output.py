# Imports.
import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import params as yamnet_params
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, regularizers 

import yamnet as yamnet_model
import params as yamnet_params
import features as features_lib
import pandas as pd
# Set random seed for TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

sr = 10000
params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
yamnet=yamnet_model.yamnet_frames_model(params)
class_names = yamnet_model.class_names('yamnet_class_map_1.csv')

#load fine-trained weights 
try:
    yamnet.load_weights('models/yamnet_ndata.h5', by_name=True)
except Exception as e:
    print(f"Error loading weights: {e}")
    sys.exit(1)

audio_path = sys.argv[1]

features = features_lib.extract_features(audio_path)
predictions = yamnet.predict(features[np.newaxis, ...])
probabilities = predictions / np.sum(predictions)
predicted_index = np.argmax(probabilities, axis=1)[0]
predicted_label = class_names[predicted_index]

# Output
# print(f"\nPredicted class index: {predicted_index}")
print(f"Predicted label: {predicted_label}")
print(f"Probabilities : {np.round(probabilities[0], 2)}")