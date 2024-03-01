import os
import glob
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, Error: {e}")
        return None
    return mfccs_processed

def load_dataset(base_path):
    features, labels = [], []
    for label in ["calls", "not calls"]:
        files = glob.glob(os.path.join(base_path, label, '*.wav'))
        for file in files:
            mfccs = extract_features(file)
            if mfccs is not None:
                features.append(mfccs)
                labels.append("yes" if label == "calls" else "no")
    return features, labels

base_dataset_path = "training-dataset"
features, labels = load_dataset(base_dataset_path)

X = np.array(features)
y = np.array(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(40,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping, model_checkpoint])

model = tf.keras.models.load_model('best_model.keras')

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy*100}%")

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

model.save('model.keras')