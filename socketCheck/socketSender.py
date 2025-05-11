import socket
import time

def start_sender():
    target_ip = "127.0.0.1"  # Set to the receiver's IP address (localhost for same machine)
    target_port = 1001       # Make sure this matches the receiver's listening port
    interval_seconds = 2      # Send a trigger every 2 seconds

    # The actual 1-bit trigger (LSB = 1)
    trigger_byte = bytes([0b00000001])  # 1-bit trigger

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP triggers to {target_ip}:{target_port} every {interval_seconds} seconds")

    try:
        while True:
            sock.sendto(trigger_byte, (target_ip, target_port))  # Send the 1-bit trigger
            print(f"Trigger sent: {trigger_byte.hex()} (as bit)")

            time.sleep(interval_seconds)  # Wait for the next trigger

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()

if __name__ == "__main__":
    start_sender()
