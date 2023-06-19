#pip install pyttsx3
#sudo apt install espeak ffmpeg libespeak1
import pyttsx3
engine = pyttsx3.init()
engine.say("Hi you! my name is lili, I like yingshaoxo.")
engine.runAndWait()