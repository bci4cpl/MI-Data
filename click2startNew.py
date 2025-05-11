"""click2start.py - Start the Brain-Computer Interface (BCI) Application

This script initializes and starts the BCI application by calling the `main()` function
from the `GUI` module. It creates the main application object, runs the GUI's main loop,
and waits for all tasks to complete before printing a message.

"""

import readLSLnew


def main():
    app = readLSLnew.main()  # get the Application object from GUI.main()
    app.experiment()

    # Wait for all tasks to complete
    print("MAIN THREAD: DONE")


if __name__ == "__main__":
    main()

"""
***
pay attention that to stop and save a fif file u need to input "n" and press enter in the console while stream run
***
"""