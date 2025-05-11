import time

from pylsl import StreamInlet, local_clock, resolve_byprop
import sys
from parameters import *
import threading
import queue

def main(keepReading, keepPulling, dataQueue):
# def main():
    """
        Main function to read data from an EEG stream and store it in a data queue in order to regulate the data
        and maintain safe stream.

        Args:
            keepReading (threading.Event): A threading event to control the reading process.
            keepPulling (threading.Event): A threading event to indicate when to read data from the stream.
            dataQueue (queue.Queue): A queue to store the EEG data samples.

        Returns:
            None
    """
    # keepReading = threading.Event()
    # keepReading.set()
    # keepPulling = threading.Event()
    # keepPulling.set()
    # dataQueue = queue.Queue()
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")

    stream_info = resolve_byprop('name', streamName, 1, 5)

    print("passed stream info")
    if len(stream_info) == 0:
        keepReading.clear()
        print("No stream found############################################################################################################")
        keepReading.clear()
        keepPulling.clear()
        sys.exit("Exiting program: No stream found.")  # Exiting the program with a message
    else:
        # create a new inlet to read from the stream
        inlet = StreamInlet(stream_info[0])
        print("Stream resolved!")

    # Saving stream meta data
    #################################################################
    # Open the file in write mode and redirect the print output to the file
    # with open(mata_data_file_path, 'w') as file:
    #     print("Stream Info:", file=file)
    #     print(inlet.info().as_xml(), file=file)
    #
    # print(f"Stream info saved to {mata_data_file_path}")
    #################################################################
    print(" HEREEE BEFORE LOOPPPPPPP")
    print(" keepReading.is_set()",keepReading.is_set())


    while keepReading.is_set():
        # Get sample
        data, timestamp_headset = inlet.pull_sample()
        timestamp = local_clock()
        # Insert sample into the dataQueue iff i need the data meaning:
        # keepPulling flag is True - the subject is shown the stimulus
        if keepPulling.is_set() and len(data) > 0:
            print(data[:])#currently its without slicing any data - if we want only electrodes data we need till row 8
            # print(len(data))
            # write to the queues
            dataQueue.put([timestamp, data])

    print("READ THREAD: DONE")


if __name__ == '_main_':
    main()