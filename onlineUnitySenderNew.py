import socket
import time
import numpy as np
from pylsl import StreamInlet, resolve_streams
import keyboard

from preprocessing import *
from dwt_svm import *
from DroneModule import Drone
# --- Real-Time Parameters ---
CHANNEL_COUNT = 8
WINDOW_DURATION = 3 # seconds
FS = 250  # Unicorn sampling rate
WINDOW_SAMPLES = int(WINDOW_DURATION * FS)



#
# print("Connected to Unicorn. Starting data stream...")
#
# try:
#     while True:
#         # Step 1: Collect a 4.5-second window of EEG data
#         print("Collecting window...")
#         eeg_buffer = np.zeros((CHANNEL_COUNT, WINDOW_SAMPLES))
#         sample_index = 0
#
#         while sample_index < WINDOW_SAMPLES:
#             samples_needed = min(32, WINDOW_SAMPLES - sample_index)  # read in chunks
#             data = device.get_data(samples_needed)  # shape: (channels, samples)
#             eeg_buffer[:, sample_index:sample_index+samples_needed] = data[:CHANNEL_COUNT, :]
#             sample_index += samples_needed
#
#         # Step 2: Preprocess
#         b_notch, a_notch = notch_filt(50, 30.0, FS)
#         eeg_notched = apply_filter(eeg_buffer, b_notch, a_notch)
#
#         b_band, a_band = butter_bandpass(4, 40, FS, order=6)
#         eeg_filtered = apply_filter(eeg_notched, b_band, a_band)
#
#         # Step 3: Feature extraction
#         eeg_window = eeg_filtered.reshape(1, CHANNEL_COUNT, WINDOW_SAMPLES)
#         features_vec = features.features_concat(eeg_window.astype(np.float64), method='dwt+csp')
#
#         # Step 4: Predict
#         prediction = svm_model.rbf_model.predict(features_vec)[0]  # 0 = left, 1 = right
#
#         # Step 5: Send via UDP
#         msg = str(prediction)
#         sock.sendto(msg.encode('utf-8'), (HOST, PORT))
#         print(f"Prediction: {msg} sent to {HOST}:{PORT}")
#
#         # Optional: Delay between windows (adjust for speed)
#         time.sleep(0.5)
#
# except KeyboardInterrupt:
#     print("Stopping real-time stream.")
#
# finally:
#     device.disconnect()
#     sock.close()
#     print("Disconnected from Unicorn and UDP socket.")
#


"----- Yoav implementation -----"


# Constants
BUFFER_SECONDS = 6
SAMPLING_RATE = 250
CHANNEL_COUNT = 8  # adjust for Unicorn hybrid black
SAMPLES_PER_BUFFER = BUFFER_SECONDS * SAMPLING_RATE
lowcut = 4
highcut = 40
order = 6
f0 = 50
Q = 30.0
distance = 40 # drone movement in cm
data_labels = []

print("Looking for EEG stream...")
streams = resolve_streams()
print(streams)
# Find the first stream of type EEG
eeg_stream = None
for stream in streams:
    print(stream.type())
    if stream.type() == 'Data':
        eeg_stream = stream
        break

if eeg_stream is None:
    raise RuntimeError("No EEG stream found.")

inlet = StreamInlet(eeg_stream)
print("EEG stream found.")

buffer = np.zeros((CHANNEL_COUNT, 0))

# UDP config
HOST = "127.0.0.1"
PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


drone = Drone()
drone.connect()
drone.is_connected()
drone.takeOff()
time.sleep(2)


print("Starting to collect data...")

while True:
    sample, timestamp = inlet.pull_sample()
    if buffer.shape[1] == 0:
        print(f"Sample shape from stream: {len(sample)}")  # e.g., 17
    if sample is not None:
        # Append sample column-wise (shape: CHANNELS x 1)
        sample_np = np.array(sample[:8]).reshape(-1, 1)
        buffer = np.hstack((buffer, sample_np))

        # If buffer is full (i.e., 4 seconds of data)
        if buffer.shape[1] >= SAMPLES_PER_BUFFER:
            print(f"\n--- New 6-second chunk ({buffer.shape[1]} samples) ---")

            # Take exactly 4 seconds worth of data
            chunk = buffer[:, :SAMPLES_PER_BUFFER]

            # Advance buffer (keep any overflow samples)
            buffer = buffer[:, SAMPLES_PER_BUFFER:]

            # Send to pipeline
            # Preprocessing
            b, a = notch_filt(f0, Q, SAMPLING_RATE)
            notched_data = apply_filter(chunk[:8, :], b, a)
            b, a = butter_bandpass(lowcut, highcut, SAMPLING_RATE, order=order)
            filtered_data = apply_filter(notched_data, b, a)
            # preprocessed = preProcess(chunk)
            # features = extractFeatures(preprocessed)
            # Feature extraction
            features = FeatureExtraction(data_labels, mode='online')
            # DWT + CSP features
            # print(filtered_data.shape)
            eeg_features = features.features_concat(filtered_data.astype(np.float64), 'dwt+csp')
            # Classifier
            svm_model = SVModel()
            prediction = svm_model.test_model_online(eeg_features)
            print(f"Predicted MI Intent: {prediction}")

            #sending command to unity
            # After you obtain `prediction`, convert it to a single integer 0 or 1 and send:
            # Option A: If `prediction` is a list or length-1 array
            prediction_value = int(prediction[0])

            # Now send over UDP:
            command = str(prediction_value)  # "0" or "1"
            sock.sendto(command.encode("utf-8"), (HOST, PORT))



            # control drone

            PREDICTION_TO_COMMAND = {
                0: "LEFT",
                1: "RIGHT",
                2: "FORWARD",
                3: "BACKWARD",
                4: "UP",
                5: "DOWN"
            }
            command_str = PREDICTION_TO_COMMAND.get(prediction_value)
            print(command_str)
            drone.move(distance, command_str)
            print(f"Drone stop in place")
            # drone.move(where="STOP_IN_PLACE")
            time.sleep(5)
            if keyboard.is_pressed('q'):
                print("Pressed Q. Landing and exiting.")
                break


# print("Land and end communication with Drone")
drone.land()

sock.close()



