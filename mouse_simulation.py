from pynput.mouse import Button, Controller
import time

mouse = Controller()

## Point the mouse to the area in the Colab interface
while True:
    mouse.click(Button.left, 1)
    time.sleep(30)  # Click every 30 seconds