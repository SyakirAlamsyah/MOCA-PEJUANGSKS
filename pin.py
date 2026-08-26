import Jetson.GPIO as io
import time

ledp = 7

io.setmode(io.BOARD)

io.setup(ledp, io.OUT, initial=io.LOW)

print("CTRL+C untuk berhenti")

try:
    while True:
        io.output(ledp, io.HIGH)
        time.sleep(1)

        io.output(ledp, io.LOW)
        time.sleep(1)

except KeyboardInterrupt:
    io.cleanup()