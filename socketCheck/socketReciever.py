import socket

def start_receiver():
    # Set the IP address and port to listen on
    host = '127.0.0.1'  # Listen on all available interfaces
    port = 1001        # Match the port with the sender's target port

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))  # Bind the socket to the address and port

    print(f"Listening for UDP packets on {host}:{port}...")

    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        print(f"Received {data.hex()} from {addr}")

if __name__ == "__main__":
    start_receiver()
