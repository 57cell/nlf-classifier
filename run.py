from tensorflow.keras.models import load_model
import librosa
import numpy as np
import pandas as pd

model = load_model('model.keras')

def extract_features(audio, sample_rate, offset, duration):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, hop_length=int(sample_rate*0.01), n_fft=int(sample_rate*0.02))
    mfccs_processed = np.mean(mfccs[:, offset:offset + duration].T, axis=0)
    return mfccs_processed

new_audio_path = 'test.wav'
audio, sample_rate = librosa.load(new_audio_path, res_type='kaiser_fast') 

segment_duration = 6
segment_overlap = 2
segments = []
segment_confidences = []

total_samples = len(audio)
segment_size = sample_rate * segment_duration
overlap_size = sample_rate * segment_overlap
num_segments = int(np.ceil((total_samples - overlap_size) / (segment_size - overlap_size)))

for i in range(num_segments):
    start_sample = int(i * (segment_size - overlap_size))
    end_sample = min(start_sample + segment_size, total_samples)
    segment_features = extract_features(audio[start_sample:end_sample], sample_rate, 0, end_sample - start_sample)
    segment_features = np.array([segment_features])
    segment_features = np.reshape(segment_features, (segment_features.shape[0], -1))
    prediction = model.predict(segment_features)
    confidence = prediction[0][0]
    start_time = start_sample / sample_rate
    end_time = end_sample / sample_rate
    segments.append((start_time, end_time, confidence))
    segment_confidences.append(confidence)

overall_confidence = np.mean(segment_confidences)

print("\nSegment Confidence Table:")
df = pd.DataFrame(segments, columns=['Start Time (s)', 'End Time (s)', 'Confidence'])
print(df)
print(f"Final confidence: {overall_confidence:.2f}")




# nothing works!!

#Segment Confidence Table:
#   Start Time (s)  End Time (s)    Confidence
#0             0.0      6.000000  1.057262e-17
#1             4.0     10.000000  3.787743e-18
#2             8.0     14.000000  9.866906e-19
#3            12.0     18.000000  1.691930e-18
#4            16.0     22.000000  1.992121e-18
#5            20.0     26.000000  8.191006e-19
#6            24.0     30.000000  2.759514e-19
#7            28.0     34.000000  5.629744e-18
#8            32.0     35.813923  9.714045e-19
#Final confidence: 0.00

# ??????????????????? how does this happen
# the test.wav used in this example was literally from the training set