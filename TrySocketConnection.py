import socket
import time

def main():
    target_ip = "127.0.0.1"   # Change to Unicorn's IP if needed
    target_port = 1000    # Change to Unicorn's listening port
    interval_seconds = 2      # Send every 2 seconds

    # This is the actual "bit" trigger — a single byte with LSB = 1
    trigger_byte = bytes([0b00000001])  # 1-bit trigger

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP triggers to {target_ip}:{target_port} every {interval_seconds} seconds")

    try:
        # Optional connection check: send an empty probe
        try:
            sock.sendto(b'', (target_ip, target_port))
            print("UDP connection check (empty packet) sent successfully.")
        except Exception as conn_ex:
            print(f"Connection check failed: {conn_ex}")
            return

        while True:
            sock.sendto(trigger_byte, (target_ip, target_port))
            print(f"[{time.strftime('%H:%M:%S')}] Trigger sent: {trigger_byte.hex()} (as bit)")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
