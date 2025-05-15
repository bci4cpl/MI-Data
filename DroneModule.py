from djitellopy import tello
from typing import Union


# TODO fix started connecting (when drone is off) and then turning on drone
# TODO rap the moving command such that the programme will not crash
class Drone:
    # Class attributes
    DEFAULT_SPEED = 30
    ALLOWED_KEYS = ["LEFT", "RIGHT", "FORWARD", "BACKWARD", "UP", "DOWN", "TURN_LEFT", "TURN_RIGHT"]
    # Constructor

    def __init__(self, speed: Union[None, int] = None):
        self.myDrone = tello.Tello()
        if speed is None:
            self.speed_of_drone = Drone.DEFAULT_SPEED  # Set default speed
        else:
            self.speed_of_drone = speed  # Set user-provided speed

    def is_connected(self):
        """
        Checks if the drone is connected.

        Returns:
            bool: True if the drone is connected, False otherwise.
        """
        try:
            battary_state = self.myDrone.get_battery()
            print(battary_state)
            is_connected = True
        except Exception as e:
            print(f"Error: {e}")
            is_connected = False

        return is_connected


    # TODO
    def connect(self):
        """
        Establishes a connection to the drone. Blocks until a successful connection is made.

        """
        max_connection_attempts = 5  # Number of connection attempts before dropping connection
        current_attempt = 1

        while current_attempt <= max_connection_attempts:
            try:
                self.myDrone.connect()
                if self.is_connected():
                    print("Drone connected successfully!")
                    return

            except Exception as e:
                print(f"Error: {e}")
                print("please connect to Drone using WiFi")

            print("Failed to connect. Retrying...")
            current_attempt += 1

    # def getInput(self, key_name: str):
        # """
        # Maps a given key name to drone movement commands.
        #
        # Args:
        #     key_name (str): The input key name representing a specific movement command.
        #
        # Returns:
        #     list: A list containing the mapped movement commands [lr, fb, ud, yv].
        #           lr (int): Left (-) or Right (+) movement value based on key_name.
        #           fb (int): Forward (+) or Backward (-) movement value based on key_name.
        #           ud (int): Up (+) or Down (-) movement value based on key_name.
        #           yv (int): Yaw (Turning) left (+) or right (-) movement value based on key_name.
        #
        # Raises:
        #     ValueError: If an invalid key_name is provided. Valid options are: ["LEFT", "RIGHT", "FORWARD",
        #                 "BACKWARD", "UP", "DOWN", "TURN_LEFT", "TURN_RIGHT"].
        # """
        # if key_name not in Drone.ALLOWED_KEYS:
        #     raise ValueError(f"Invalid keyName: {key_name}. Allowed options are: {Drone.ALLOWED_KEYS}")
        #
        # lr, fb, ud, yv = 0, 0, 0, 0
        # if key_name == "0" or key_name == "LEFT": lr = -Drone.DEFAULT_SPEED
        # elif key_name == "1" or key_name == "RIGHT": lr = Drone.DEFAULT_SPEED
        # if key_name == "FORWARD": fb = Drone.DEFAULT_SPEED
        # elif key_name == "BACKWARD": fb = -Drone.DEFAULT_SPEED
        # if key_name == "UP": ud = Drone.DEFAULT_SPEED
        # elif key_name == "DOWN": ud = -Drone.DEFAULT_SPEED
        # if key_name == "TURN_LEFT": yv = Drone.DEFAULT_SPEED
        # elif key_name == "TURN_RIGHT": yv = -Drone.DEFAULT_SPEED
        #
        # return [lr, fb, ud, yv]
    def getInput(self, distance, key_name: str):
        """
            Maps a given key name to drone movement commands.

            Args:
                key_name (str): The input key name representing a specific movement command.

            Returns:
                list: A list containing the mapped movement commands [lr, fb, ud, yv].
                      lr (int): Left (-) or Right (+) movement value based on key_name.
                      fb (int): Forward (+) or Backward (-) movement value based on key_name.
                      ud (int): Up (+) or Down (-) movement value based on key_name.
                      yv (int): Yaw (Turning) left (+) or right (-) movement value based on key_name.

            Raises:
                ValueError: If an invalid key_name is provided. Valid options are: ["LEFT", "RIGHT", "FORWARD",
                            "BACKWARD", "UP", "DOWN", "TURN_LEFT", "TURN_RIGHT"].
        """
        if key_name not in Drone.ALLOWED_KEYS:
            raise ValueError(f"Invalid keyName: {key_name}. Allowed options are: {Drone.ALLOWED_KEYS}")

        lr, fb, ud, yv = 0, 0, 0, 0
        if key_name == "LEFT": self.myDrone.move_left(distance)
        elif key_name == "RIGHT": self.myDrone.move_right(distance)
        if key_name == "FORWARD": fb = Drone.DEFAULT_SPEED
        elif key_name == "BACKWARD": fb = -Drone.DEFAULT_SPEED
        if key_name == "UP": ud = Drone.DEFAULT_SPEED
        elif key_name == "DOWN": ud = -Drone.DEFAULT_SPEED
        if key_name == "TURN_LEFT": yv = Drone.DEFAULT_SPEED
        elif key_name == "TURN_RIGHT": yv = -Drone.DEFAULT_SPEED

        return [lr, fb, ud, yv]


    # TODO

    def takeOff(self):
        self.myDrone.takeoff()

    def land(self):
        self.myDrone.land()

    def move(self, distance, where):
        command = self.getInput(distance, key_name=where)
        self.myDrone.send_rc_control(command[0], command[1], command[2], command[3])

    def stop(self):
        # Immediately stops all movement (hover)
        self.myDrone.send_rc_control(0, 0, 0, 0)

    # def getInput(self, keyName):
    #
    #     lr, fb, ud, yv = 0, 0, 0, 0
    #
    #     speed = 50
    #     if keyName == "LEFT":
    #         lr = -speed
    #     elif keyName == "RIGHT":
    #         lr = speed
    #     if keyName == "UP":
    #         fb = speed
    #     elif keyName == "DOWN":
    #         fb = -speed
    #     if keyName == "w":
    #         ud = speed
    #     elif keyName == "s":
    #         ud = -speed
    #     if keyName == "a":
    #         yv = speed
    #     elif keyName == "d":
    #         yv = -speed
    #
    #     return [lr, fb, ud, yv]
