import glob
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datetime import datetime, timedelta

# Define feature extraction function
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=None, mono=True)
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    return mfccs, chroma, mel, contrast, tonnetz

# Define audio parsing function
def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    ignored = 0
    features, labels = np.empty((0, 161)), np.empty(0)
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_features(fn)
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                label = 1 if 'frog' in fn else 0  # Adjust labeling logic as per dataset
                labels = np.append(labels, [label] * mfccs.shape[0])
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                ignored += 1
                print(f"Ignored file: {fn} due to error: {e}")
    print("Ignored files:", ignored)
    return np.array(features), np.array(labels, dtype=np.int)

# One-hot encoding function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Load or extract features and labels
parent_dir = 'NLF_dataset/audio'  # Adjust dataset path
sub_dirs = ['positive', 'negative']  # Adjust folder names as per your dataset

try:
    labels = np.load('labels.npy')
    features = np.load('features.npy')
    print("Features and labels found!")
except:
    print("Extracting features...")
    features, labels = parse_audio_files(parent_dir, sub_dirs)
    np.save('features.npy', features)
    np.save('labels.npy', labels)

labels = one_hot_encode(labels)

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(train_x)
train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

# TensorFlow model parameters
training_epochs = 5000
n_dim = features.shape[1]
n_classes = 2
n_hidden_units_one = 256
n_hidden_units_two = 256
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01
model_path = "model"

# TensorFlow placeholders and variables
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

# Initialize variables and TensorFlow Saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Define cost function and optimizer
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Define accuracy metrics
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train the model
batch_size = 10000
patience_cnt = 0
patience = 16
min_delta = 0.01
stopping = 0

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        if stopping == 0:
            total_batch = int(train_x.shape[0] / batch_size)
            train_x = shuffle(train_x, random_state=42)
            train_y = shuffle(train_y, random_state=42)
            for i in range(total_batch):
                batch_x = train_x[i * batch_size:i * batch_size + batch_size]
                batch_y = train_y[i * batch_size:i * batch_size + batch_size]
                _, cost = sess.run([optimizer, cost_function], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, cost)
            if epoch % 100 == 0:
                print("Epoch:", epoch, "cost", cost)
            if epoch > 0 and abs(cost_history[epoch - 1] - cost_history[epoch]) > min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                print("Early stopping at epoch", epoch, ", cost", cost)
                stopping = 1

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y, 1))
    save_path = saver.save(sess, model_path)
    print("Model saved at:", save_path)

p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
print("F-Score:", f)
print("Precision:", p)
print("Recall:", r)

# Function to extract frog detections from a 6-hour audio clip
def detect_frog_calls(audio_file, model_path, output_dir):
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        X_ph = tf.placeholder(tf.float32, [None, n_dim])
        y_pred_ph = tf.argmax(tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(X_ph, W_1) + b_1), W_2) + b_2), W) + b), 1)
        y_conf_ph = tf.reduce_max(tf.nn.softmax(tf.matmul(tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(X_ph, W_1) + b_1), W_2) + b_2), W) + b), axis=1)

        audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        start_time = datetime(2024, 4, 23, 0, 0)  # Adjust date and starting time
        window_size = 60  # 1 minute window
        step_size = window_size // 2  # 30 seconds overlap
        window_samples = window_size * sample_rate
        step_samples = step_size * sample_rate

        os.makedirs(output_dir, exist_ok=True)

        for i in range(0, len(audio_data) - window_samples, step_samples):
            window_data = audio_data[i:i + window_samples]
            mfccs, chroma, mel, contrast, tonnetz = extract_features_from_array(window_data, sample_rate)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            ext_features = sc.transform([ext_features.flatten()])

            prediction, confidence = sess.run([y_pred_ph, y_conf_ph], feed_dict={X_ph: ext_features})

            if prediction[0] == 1 and confidence[0] > 0.5:
                detected_time = start_time + timedelta(seconds=(i / sample_rate))
                timestamp_str = detected_time.strftime("%Y-%m-%d_%H-%M")
                output_file = os.path.join(output_dir, f"{timestamp_str}.wav")

                # Extract the entire 1-minute window
                librosa.output.write_wav(output_file, window_data, sample_rate)
                print(f"Detected frog call at {timestamp_str} with confidence {confidence[0]:.2f}")

# Function to extract features from an audio array
def extract_features_from_array(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    return mfccs, chroma, mel, contrast, tonnetz

# Example usage of the detection function
audio_file = "6_hour_audio_clip.wav"  # Adjust path to the 6-hour audio clip
output_dir = "frog_detections"
detect_frog_calls(audio_file, model_path, output_dir)
