#pip install pywin32
import win32com.client as wincom

speaker = wincom.Dispatch("SAPI.SpVoice")
speaker.Speak("Hi you! my name is lili, I like yingshaoxo")