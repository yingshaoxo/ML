#pip install pyttsx3
#sudo apt install espeak ffmpeg libespeak1
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')  # getting details of current voice
engine.setProperty('voice', voice[1].id)  # this is female voice
engine.say("Hi you! my name is lili, I like yingshaoxo.")
engine.runAndWait()

