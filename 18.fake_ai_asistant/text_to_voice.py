#pip install pyttsx3
#sudo apt install espeak ffmpeg libespeak1
import pyttsx3
engine = pyttsx3.init()

def say_somthing(text: str):
    #engine.say("Hi you! my name is lili, I like yingshaoxo.")
    engine.say(text)
    engine.runAndWait()