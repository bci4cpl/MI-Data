import time
import mne
from parameters import *
import random
from datetime import datetime
# from cleanPsychopyLOG import main as cleanLOG
# from featureExtraction import main as extractFeatures
# from preProcessing import main as preProcess
# from Model import main as call_model
import keyboard
import os
import pandas as pd
import sys
# from validationFiles.modelCompetition2_0 import *
# from psychopy import logging, core, visual
# import pylsl
# from DroneModule import Drone
# import fitPlus


#TODO need to change function name to folder and not file
def createFile(state):
    """
    Create a directory for the experiment or test session.

    Args:
        state (str): Either "TRAIN" for training or "TEST" state for testing.

    Returns:
        str: The path to the created directory.
        str: The name of the directory.
    """
    # date & time
    date = datetime.now().strftime("%d_%m_%Y at %I_%M_%S_%p")

    # Create the "EXP_{date}" directory
    if state == "TRAIN":
        exp_dir = f"output_files/EXP_{date}/"
    else:
        exp_dir = f"output_files/test_{date}/"
    os.makedirs(exp_dir, exist_ok=True)
    folder_name = f"EXP_{date}"
    return exp_dir, folder_name


def getfifFilesList(main_folder):
    # get all the experiment folders
    fifFolder = [f for f in os.listdir(main_folder) if
                 os.path.isfile(os.path.join(main_folder, f)) and f.endswith(".fif")]
    return fifFolder


def clearQueue(dataQueue):
    """
    Clear items from a queue.

    Args:
        dataQueue (queue.Queue): The queue to be cleared.
    """
    while not dataQueue.empty():
        dataQueue.get()


# class Timer:
#     def getTime(self):
#         return pylsl.local_clock()



# def handleLOG(exp_path):
    """
    Handle the Psychopy experiment logs.

    Args:
        exp_path (str): The path to the experiment directory.
    """

    # logging.flush()
    # cleanLOG(exp_path)
    # return


def saveData(exp_path, raw, timeStampsOutputDF):
    """
    Save experiment data.

    Args:
        exp_path (str): The path to the experiment directory.
        raw (mne.io.RawArray): The MNE raw data.
        timeStampsOutputDF (pd.DataFrame): The timestamps data.
    """

    date = datetime.now().strftime("%d_%m_%Yat%I_%M_%S_%p")
    file_path = exp_path + raw_data_folder_name
    createFolder(file_path)
    nameOfRawFile = "data_of_" + date + ".fif"
    raw.save(file_path + nameOfRawFile)
    timeStampsOutputDF.to_csv(file_path + "timestamps_" + date + ".csv")
    return





def createMNERawObject(dataQueue):
    """
        Create MNE Raw object from data in a queue.

        Args:
            dataQueue (queue.Queue): The data queue.

        Returns:
            mne.io.RawArray: The MNE raw data.
            pd.DataFrame: Timestamps data.
    """

    # create raw file using the data in the dataQueue
    # we need to correctly adjust the ch_names and ch_types to our relevent electrodes and matrix len and shape
    listOfData = list()
    listOfTimeStamp = list()
    while not dataQueue.empty():
        currDataSample = dataQueue.get()
        listOfTimeStamp.append(currDataSample[0])
        listOfData.append(currDataSample[1])

    # print(listOfData)
    # print("len",len(listOfData))
    if listOfData is not None and len(listOfData) > 0:
        dataOutputDF = pd.DataFrame(listOfData)
        timeStampsOutputDF = pd.DataFrame(listOfTimeStamp)
        #here we need to define info from createMNEobjectStructure
        dataOutputDF = np.concatenate((dataOutputDF.values.T[:8], np.array([dataOutputDF.values.T[16]])), axis=0)
        # dataOutputDF = dataOutputDF.values.T[:9]
        info = mne.create_info(ch_names=ch_types, sfreq=samplingRate, ch_types=ch_types)
        raw = mne.io.RawArray(dataOutputDF, info) # TODO - understand which channel is trigger!
    else:
        raise Exception("something is wrong with the save of the recording.......")

    return raw, timeStampsOutputDF


def showExperiment(state, keepReading, keepPulling, dataQueue, test_model=None, curr_drone_status=0, drone=None):
    """
        Run the motor imagery experiment.

        Args:
            state (str): Either "TRAIN" for training or "TEST" state for testing.
            keepReading (threading.Event): Event to control data reading - communication with the headset (Atomic boolean value).
            keepPulling (threading.Event): Event to control data pulling - pulling from the dataQueue (that regulates the data stream), (Atomic boolean value).
            dataQueue (queue.Queue): The data queue.
            test_model (str): The test model to use for predictions.
    """
    print("STARTED EXPERIMENT SCRIPT")
    # create the proper folder for the exp / test
    exp_path, folder_name = createFile(state)

    # set up logger
    # logging.LogFile(fileName, level=logging.EXP, filemode='w')
    # studyClock = Timer()
    # logging.setDefaultClock(studyClock)  # this is the logger

    # indicate where there is need to read from the dataQueue
    pullFromQueue = True



    if not keepReading.is_set():
        sys.exit("Exiting program: No stream found.")  # Exiting the program with a message

    if state == "TRAIN":
        # Instructions for TRAIN
        print("before activating pull")
        # Indicate that relevant data is coming
        keepPulling.set()
        print("after activating pull curr situation ", keepPulling.is_set())
       # time.sleep(4)  # to remove the first trials being noise on calibration
        # Show the experiment
        # count = 2
        while pullFromQueue:
            user_input = input("Press 'y' (ignore), press 'n' to end: ").lower()

            if user_input == 'y':
                return True  # Yes
            elif user_input == 'n':
                pullFromQueue = False  # No
            else:
                print("Invalid input. Please press 'y' or 'n'.")
            # maybe create c# flag from the simulator
            # finished with the experiment
            # if count == 0:
            #     pullFromQueue = False
        # killing the threads
        # wait for the interval not to be too short
        time.sleep(0.7)
        keepPulling.clear()
        print("keepPulling.value", keepPulling.is_set())
        keepReading.clear()
        print("keepReading.value", keepReading.is_set())

        # creating MNE object
        raw, timeStampsOutputDF = createMNERawObject(dataQueue)

        # save the data
        saveData(exp_path, raw, timeStampsOutputDF)

        # the rest of the pipeline

        # take care of the log - before preprocessing
        # handleLOG(exp_path)


    else:  # test condition
        # Check drone status
        print("curr_drone_status is :", curr_drone_status)
        if curr_drone_status == 1:
            # optional
            drone.takeOff()

        # instruction for TEST
        print("before activating pull")
        # Indicate that relevant data is coming
        keepPulling.set()
        print("after activating pull curr situation ", keepPulling.is_set())
        # time_for_clean_recording = 1.5
        # time.sleep(time_for_clean_recording) # to remove the first trials being noise on calibration
        # init of raw_queue list
        raw_queue = list()
        timeStamp_queue = list()
        test_index = 0
        listOfLabels = list()
        listOfPreds = list()
        counter_left = 0
        counter_right = 0
        # make sure you send model when clicking "testModel"
        # while not keyboard.is_pressed('esc'):
        # while pullFromQueue:
        # time.sleep(1)
        print("___________________________________ TEST STARTED ____________________________________________")
        # Indicate that relevant data is coming
        # keepPulling.set()

        keepPulling.clear()
        print("keepPulling status: ", keepPulling.is_set())

        # if curr_drone_status == 1:
        #     drone.keep_drone_alive()

        # create raw file
        raw, timeStampsOutputDF = createMNERawObject(dataQueue)
        # clear the queue in order for us to have ONLY the relevant information - test only
        # SUPPOSE TO BE EMPTY!!!!!!!
        clearQueue(dataQueue)

        # add raw object to the raw_queue
        raw_queue.append(raw)
        timeStamp_queue.append(timeStampsOutputDF)

        # the rest of the pipeline
        # take care of the log
        # handleLOG(exp_path)

        if curr_drone_status == 1:
            prediction_direction = listOfPreds[0].split(",")[0]
            print(f"Drone rotate :{prediction_direction}")
            drone.move(where="TURN_" + prediction_direction)
            time.sleep(3)
            print(f"Drone stop in place")
            drone.move(where="STOP_IN_PLACE")
            time.sleep(1)
            print("Land and end communication with Drone")
            drone.end_drone()

        # if test_index == test_num_of_epochs:
        #     break
        # repeat

        # testing is over
        print("testing is over")
        # present overall scores?

        # save the raw_queue object into a folder
        if len(raw_queue) != len(timeStamp_queue):
            print("Error: The raw_queue and the timeStamp_queue lists have different lengths.")
        else:
            # saves at the same time my overwrite the raw files
            # for now solving this with sleep
            # test_index_for_save - add it to the name and problem solved
            for currRow, currTS in zip(raw_queue, timeStamp_queue):
                saveData(exp_path, currRow, currTS)
                time.sleep(1)

        file_path = exp_path + 'comparing_model_results.txt'
        with open(file_path, 'w') as file:
            file.write("labels, prediction on the model in test\n")
            # Step 2: Use a loop to write elements from both lists
            for item1, item2 in zip(listOfLabels, listOfPreds):
                file.write(f'{item1}, {item2}\n')

    # Adding fitPlus for the pipline
    # fitPlus.main()
    # print("fitPlus has finished")

    # move test folder to total_folder folder
    # moveFile()

    # Killing reading thread
    keepReading.clear()

    # Close the PsychoPy application completely
    # core.quit()
    # if curr_drone_status == 1:
    #    try:
    #        drone.end_drone()
    #    except:
    #        print("DRONE HAS ALREADY BEEN ENDED")
    print("EXPERIMENT: DONE")
