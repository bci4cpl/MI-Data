import mne
import numpy as np
import UnicornPy

# from pylsl import StreamInlet, resolve_streams
# import pandas as pd
#
# # initialize the streaming layer
# finished = False
# streams = resolve_streams()
# inlet = StreamInlet(streams[0])
#
# # initialize the colomns of your data and your dictionary to capture the data.
# columns = ['Time', 'FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'AccX', 'AccY', 'AccZ',
#            'Gyro1', 'Gyro2', 'Gyro3', 'Battery', 'Counter', 'Validation']
# data_dict = dict((k, []) for k in columns)
#
# while not finished:
#     # get the streamed data. Columns of sample are equal to the columns variable, only the first element being timestamp
#     # concatenate timestamp and data in 1 list
#     data, timestamp = inlet.pull_sample()
#     all_data = [timestamp] + data
#
#     # updating data dictionary with newly transmitted samples
#     i = 0
#     for key in list(data.keys()):
#         data_dict[key].append(all_data[i])
#         i = i + 1
#
#     # data is collected at 250 Hz. Let's stop data collection after 60 seconds. Meaning we stop when we collected 250*60 samples.
#     if len(data_dict['Time']) >= 250 * 60:
#         finished = True
#
# # lastly, we can save our data to a CSV format.
# data_df = pd.DataFrame.from_dict(data_dict)
# data_df.to_csv('EEGdata.csv', index=False)

ch_list = ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'Validation Indicator']
fs = UnicornPy.SamplingRate
number_of_scans = 1000000 # The number of scans we want to read, adjust this based on your needs
serial_number = UnicornPy.GetAvailableDevices(True)[0] # serial number of the unicorn headset
unicorn_device = UnicornPy.Unicorn(serial_number)
config = unicorn_device.GetConfiguration() # Get the current configuration
for i in range(8, 16):  # Channel 9 (index 8) to Channel 16 (index 15)
    channel = config.Channels[i]
    channel.Enabled = False  # Disable the channel
unicorn_device.SetConfiguration(config) # Apply the updated configuration


# Start acquisition before fetching data
unicorn_device.StartAcquisition(False)  # Set to True for test signal mode
number_of_channels = unicorn_device.GetNumberOfAcquiredChannels() # get the number of channels
buffer_length = number_of_scans * number_of_channels * 4 # Allocate a destination buffer for the data (1 float per channel)
# destination_buffer = bytearray(buffer_length)  # 4 bytes per 32-bit float
destination_buffer = np.zeros(buffer_length // 4, dtype=np.float32)  # Allocate as numpy array
unicorn_device.GetData(number_of_scans, destination_buffer, buffer_length) # Read the data into the buffer
print(f"Retrieved {number_of_scans} scans with {number_of_channels} channels.") # Process and print the data

path = r'C:\Users\owner\PycharmProjects\PythonProject\MI-ErrP\output_files\EXP_30_12_2024 at 07_27_19_PM\Raw\data_of_30_12_2024at07_28_40_PM.fif'
dataset = mne.io.read_raw_fif(path)
eeg_raw_data = dataset.get_data()
print('y')