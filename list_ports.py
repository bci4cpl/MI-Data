# list_ports.py
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for p in ports:
    print(f"{p.device}\t– {p.description}")
