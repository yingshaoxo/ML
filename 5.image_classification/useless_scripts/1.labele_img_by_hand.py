from PIL import Image
from auto_everything.base import Terminal
from time import sleep
import os

t = Terminal()
t.run("sudo apt-get install imagemagick")


from pynput.keyboard import Listener as KeyboardListener
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController
from pynput.mouse import Button
from pynput.keyboard import Key

mouseControl = MouseController()
keyboardControl = KeyboardController()


def on_release(key):
    try:
        k = key.char
    except:
        k = key.name
    if k == "1" or k == "0":
        sleep(0.3)
        keyboardControl.press(Key.enter)
        keyboardControl.release(Key.enter)


keyListen = KeyboardListener(on_release=on_release)
keyListen.start()


def do():
    unlabeled_picture_dir = 'dataset/unlabeled_images/'
    for file_name in os.listdir(unlabeled_picture_dir):
        try:
            origin_path = unlabeled_picture_dir + file_name
            img = Image.open(origin_path).convert('RGBA')
            img.show()

            sleep(0.3)
            mouseControl.press(Button.left)
            mouseControl.release(Button.left)

            choice = int(input("yes or not?(1 or 0) "))
            t.run_command("pkill display")
            if choice == 1:
                os.rename(origin_path, 'dataset/training_set/cats/' + file_name)
                #os.rename(origin_path, 'dataset/test_set/cats/' + file_name)
            else:
                os.rename(origin_path, 'dataset/training_set/dogs/' + file_name)
                #os.rename(origin_path, 'dataset/test_set/cats/' + file_name)
        except Exception as e:
            print(e)


do()
