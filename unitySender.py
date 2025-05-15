import socket

HOST = "127.0.0.1"  # Unity machine IP (localhost for same PC)
PORT = 5005         # Must match Unity port

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Type 0 (left) or 1 (right), then Enter. 'q' to quit.")
while True:
    cmd = input("> ")
    if cmd == 'q':
        break
    if cmd in ('0','1'):
        sock.sendto(cmd.encode('utf-8'), (HOST, PORT))
    else:
        print("Invalid—use 0, 1, or q.")
