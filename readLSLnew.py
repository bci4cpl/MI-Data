"""GUI.py - Graphical User Interface (GUI) for the Brain-Computer Interface (BCI) Application

This script defines the GUI for the BCI application. It provides functionality to record real or fake EEG data,
run experiments, train models, and test models offline or in real-time.

"""

import threading
import queue
import sys
from readLslStream import main as sendReal
from concurrent.futures import ThreadPoolExecutor
from ExperimentNew import showExperiment as activateExperiment
from sanity_check_experiment import main as eyes_open_eyes_closed
from featureExtraction import main as featureExtractionMain
from PyQt5.QtWidgets import QFileDialog, QAbstractItemView, QListView, QTreeView, QApplication, QDialog, QLabel, QVBoxLayout
from joblib import load, dump
from parameters import *
from testModelOffLine import main as testModelOffline
# from DroneModule import Drone



# folder selector
class getExistingDirectories(QFileDialog):
    def __init__(self, label_text, *args):
        super(getExistingDirectories, self).__init__(*args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.Directory)
        self.setOption(self.ShowDirsOnly, True)
        self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Add the dynamically provided label text to the layout
        layout = self.layout()
        instruction_label = QLabel(label_text)
        layout.addWidget(instruction_label)


class App:
    """
        Main application class for the MI trainer GUI.
    """


    def __init__(self):
        #super().__init__()

        # params configuration
        self.keepReading = threading.Event()
        self.keepReading.set()
        self.keepPulling = threading.Event()
        self.keepPulling.clear()
        # init thread pool
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.readFromLSLStream = None
        self.pullFromLSLStream = None
        self.generateLSLStream = None
        # concurrent Queues for the data and the time stamps
        self.dataQueue = queue.Queue()


    def experiment(self):
        """
            Start the MI Experiment based on user selections.
        """
        #defining the way - if to pull or stop - the data from the EEG
        # if self.radio_var.get() == 1:

        print("Read REAL data")
        self.readFromLSLStream = self.executor.submit(sendReal, self.keepReading, self.keepPulling, self.dataQueue)

        if self.keepReading.is_set():
            print("Stream resolved - Experiment activated")
            activateExperiment("TRAIN", self.keepReading, self.keepPulling, self.dataQueue, None)

        else:
            print("Did not resolve stream - Experiment was not activated\n\n Exiting\n\n\n\n")
        return

    def trainModelOffLine(self):
        # choose what files you want to train on - and let the pipline flow
        """
            Train the model offline using selected folders.
        """
        qapp = QApplication(sys.argv)
        dlg = getExistingDirectories("                                      Please select folders to train on.....")
        selected_folders_to_train_on = list()
        if dlg.exec_() == QDialog.Accepted:
            print("Train on: ")
            selected_folders_to_train_on = dlg.selectedFiles()
            print(selected_folders_to_train_on)
        # if len(selected_folders_to_train_on) == 0:
        #     print("No folders to train on!")
        #     exit(1)
        featureExtractionMain("TRAIN", selected_folders_to_train_on=selected_folders_to_train_on, equalizeEpochs=True)
        # the model will be saved in the proper folder
        return


    def testModelOffLine(self):
        """
                Train and test the model offline using selected folders.
        """
        # TODO - prevent data leakage by avoiding training and testing on same set
        # TODO - add labels to the window - what kind of file should i choose

        self.trainModelOffLine()
        # load model - from known path
        test_model_path = output_files + models_folder_name + model_filename
        try:
            test_model = load(test_model_path)
        except FileNotFoundError:
            print("There is no model ready for testing in the default directory")
            return

        # choose folders to test on
        qapp = QApplication(sys.argv)
        dlg = getExistingDirectories(label_text="CTkScrollableFrame")
        selected_folders_to_test_on = list()
        if dlg.exec_() == QDialog.Accepted:
            print("Test on: ")
            selected_folders_to_test_on = dlg.selectedFiles()
            print(selected_folders_to_test_on)
        # test model offline - ASSUMING LABEL TEXT FILES
        testModelOffline(selected_folders_to_test_on, test_model)

    def sanity_check_experiment(self):

        print("Read REAL data")
        self.readFromLSLStream = self.executor.submit(sendReal, self.keepReading, self.keepPulling, self.dataQueue)


        eyes_open_eyes_closed("sanity_check", self.keepReading, self.keepPulling, self.dataQueue, test_model=None)
        return

    def modifiedDestroy(self):
        """
            Custom function to handle application cleanup.
        """
        self.keepPulling.clear()
        self.keepReading.clear()
        if self.pullFromLSLStream is not None:
            self.pullFromLSLStream.result()
        if self.readFromLSLStream is not None:
            self.keepReading.clear()
            self.readFromLSLStream.result()
        if self.generateLSLStream is not None:
            self.generateLSLStream.result()
        # self.quit()


def main():
    """
        Main function to create and run the MI trainer GUI application.

        Returns:
            App: The main application instance.
    """
    app = App()

    return app


if __name__ == "__main__":
    app = App()
    app.mainloop()
