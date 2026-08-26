import Jetson.GPIO as io
import time

# ledp = 7

# io.setmode(io.BOARD)

# io.setup(ledp, io.OUT, initial=io.LOW)

# print("CTRL+C untuk berhenti")

# try:
#     while True:
#         io.output(ledp, io.HIGH)
#         time.sleep(2)

#         io.output(ledp, io.LOW)
#         time.sleep(2)

# except KeyboardInterrupt:
#     io.output(ledp, io.LOW)
#     io.cleanup()

buzp = 7

io.setmode(io.BOARD)

io.setup(buzp, io.OUT, initial=io.LOW)

try:
    print("Buzzer menyala, tekan Ctrl+C untuk berhenti.")
    while True:
        # Nyalakan buzzer
        io.output(buzp, io.HIGH)
        time.sleep(0.5)
        
        # Matikan buzzer
        io.output(buzp, io.LOW)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nProgram berhenti.")
finally:
    io.output(buzp, io.LOW)
    io.cleanup()

