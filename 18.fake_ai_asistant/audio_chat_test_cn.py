from datetime import time
from time import sleep
import json
from typing import Any
import os
from pprint import pprint

from auto_everything.ml import Yingshaoxo_Text_Generator
from auto_everything.terminal import Terminal
terminal = Terminal()

yingshaoxo_text_generator = Yingshaoxo_Text_Generator(
    input_txt_folder_path="/home/yingshaoxo/CS/ML/18.fake_ai_asistant/input_txt_files",
    use_machine_learning=False
)

def decode_response(text: str, chat_context: str):
    #print("`"+text+"`")
    splits = text.split("\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n")
    if (len(splits) > 1):
        response = splits[1].strip()
    elif (len(splits) == 1):
        response = splits[0].strip()
    else:
        response = ""
    new_code = f"""
chat_context = '''{chat_context}'''

{response}
"""
    final_response = terminal.run_python_code(code=new_code)
    if final_response.strip() == "":
        final_response = response
    final_response = "\n".join([one for one in final_response.split("\n") if not one.strip().startswith("__**")])
    return final_response



class Speech_Recognizer():
    def __init__(self, language: str = 'en'):
        # pip install vosk
        # pip install sounddevice
        import queue
        import sys
        import sounddevice
        from vosk import Model, KaldiRecognizer
        from auto_everything.time import Time

        self.queue = queue
        self.sys = sys
        self.sounddevice = sounddevice
        self.time_ = Time()

        if language == "en":
            self.vosk_model = Model(lang="en-us")
        else:
            self.vosk_model = Model(model_name="vosk-model-cn-0.22")

        self.KaldiRecognizer = KaldiRecognizer

        self.microphone_bytes_data_queue = queue.Queue()
    
    def recognize_following_speech(self, timeout_in_seconds: int | None = None) -> str:
        while self.microphone_bytes_data_queue.empty() == False:
            self.microphone_bytes_data_queue.get_nowait()

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=self.sys.stderr)
            self.microphone_bytes_data_queue.put(bytes(indata))

        try:
            device_info = self.sounddevice.query_devices(None, "input")
            samplerate = int(device_info["default_samplerate"]) #type:ignore
                
            with self.sounddevice.RawInputStream(samplerate=samplerate, blocksize = 8000, device=None,
                    dtype="int16", channels=1, callback=callback):
                rec = self.KaldiRecognizer(self.vosk_model, samplerate)

                start_time = self.time_.get_current_timestamp_in_10_digits_format()
                while True:
                    data = self.microphone_bytes_data_queue.get()
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result())["text"] #type:ignore
                        text = text.replace(" ", "").strip()
                        if len(text) != 0:
                            #print(text)
                            return text
                    else:
                        # print(rec.PartialResult())
                        pass
                    end_time = self.time_.get_current_timestamp_in_10_digits_format()
                    if timeout_in_seconds != None:
                        duration = self.time_.get_datetime_object_from_timestamp(end_time) - self.time_.get_datetime_object_from_timestamp(start_time)
                        if duration.seconds > timeout_in_seconds:
                            return ""
        except Exception as e:
            print(e)
            return ""


class Yingshaoxo_Translator():
    def __init__(self):
        # pip install dl-translate
        import dl_translate
        from auto_everything.language import Language
        self.dl_translate = dl_translate
        self.dl_translate_model = self.dl_translate.TranslationModel(device="cpu")
        self.languages = self.dl_translate.lang
        self._language = Language()
    
    def translate(self, text: str, from_language: Any, to_language: Any, sentence_seperation: bool = False) -> str:
        try:
            text = text.strip()
            if sentence_seperation == True:
                data_list = self._language.seperate_text_to_segments(text=text, ignore_space=True)
                """
                [
                    {
                        "is_punctuation_or_space": true, "text": "?",
                    }, {
                        "is_punctuation_or_space": false, "text": "Yes",
                    },
                ]
                """
                text_list = []
                for segment in data_list:
                    if segment["is_punctuation_or_space"] == False:
                        result = self.dl_translate_model.translate(segment["text"], source=from_language, target=to_language)
                        result = str(result).strip("!\"#$%&'()*+, -./:;<=>?@[\\]^_`{|}~ \n，。！？；：（）［］【】")
                        text_list.append(result)
                    else:
                        text_list.append(segment["text"])
                return "".join(text_list)
            else:
                return self.dl_translate_model.translate(text, source=from_language, target=to_language) #type: ignore
        except Exception as e:
            print(e)
            return text
    
    def chinese_to_english(self, text: str, sentence_seperation: bool = False):
        return self.translate(text=text, from_language=self.languages.CHINESE, to_language=self.languages.ENGLISH, sentence_seperation=sentence_seperation)

    def english_to_chinese(self, text: str, sentence_seperation: bool = False):
        return self.translate(text=text, from_language=self.languages.ENGLISH, to_language=self.languages.CHINESE, sentence_seperation=sentence_seperation)


class Yingshaoxo_Text_to_Speech():
    def __init__(self):
        #pip install TTS
        #sudo apt install ffmpeg                 or          https://github.com/markus-perl/ffmpeg-build-script#:~:text=maintain%20different%20systems.-,Installation,-Quick%20install%20and
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        from TTS.api import TTS
        self.TTS = TTS

        from auto_everything.terminal import Terminal
        from auto_everything.disk import Disk
        self.terminal = Terminal()
        self.disk = Disk()

        import torch
        # use_gpu = True if torch.cuda.is_available() else False
        use_gpu = False
        self.torch = torch

        #self.tts_en = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=use_gpu)
        self.tts_en = TTS("tts_models/en/ljspeech/fast_pitch", gpu=use_gpu)
        self.tts_cn = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST", gpu=use_gpu)

    def _language_splitor(self, text: str):
        language_list = []
        index = 0
        while True:
            temp_string = ""
            if (index >= len(text)):
                break
            char = text[index]
            while ord(char) < 128:
                # english
                char = text[index]
                temp_string += char
                index += 1
                if (index >= len(text)):
                    break
            if (temp_string.strip() != ""):
                temp_string = temp_string[:-1]
                index -= 1
                language_list.append({
                    "language": "en",
                    "text": temp_string
                })

            temp_string = ""
            if (index >= len(text)):
                break
            char = text[index]
            while not ord(char) < 128:
                # chinese 
                char = text[index]
                temp_string += char
                index += 1
                if (index >= len(text)):
                    break
            if (temp_string.strip() != ""):
                temp_string = temp_string[:-1]
                index -= 1
                language_list.append({
                    "language": "cn",
                    "text": temp_string
                })

            if (index+1 >= len(text)):
                break

        if len(language_list) > 0:
            language_list[-1]["text"] += text[-1]
        
        new_list = []
        for index, one in enumerate(language_list):
            new_text = language_list[index]["text"].strip()
            if len(new_text) > 0:
                new_list.append({
                    'language': one['language'],
                    'text': new_text
                })

        return new_list

    def _speak_it(self, language: str, text: str):
        output_file = os.path.abspath(os.path.join(self.disk.get_a_temp_folder_path(), "output.wav"))
        self.disk.create_a_folder(self.disk.get_directory_path(output_file))

        if (language == "en"):
            tts = self.tts_en
        else:
            tts = self.tts_cn

        try:
            if tts.speakers == None:
                tts.tts_to_file(text=text, file_path=output_file)
            else:
                tts.tts_to_file(text=text, file_path=output_file, speaker=tts.speakers[0], language=tts.languages[0], speed=2.5)
        except Exception as e:
            print(e)

        self.terminal.run(f"""
        ffplay -autoexit -nodisp "{output_file}"
                """, wait=True)
        
        self.disk.delete_a_file(output_file)

    def speak_it(self, text: str):
        data_ = self._language_splitor(text)
        for one in data_:
            print(one)
            text = one["text"]
            # if one["language"] == "en":
            #     self._speak_it(language=one["language"], text=text + ".")
            # else:
            #     self._speak_it(language=one["language"], text=text + "。")
            text = text.strip("!\"#$%&'()*+, -./:;<=>?@[\\]^_`{|}~ \n，。！？；：（）［］【】")
            self._speak_it(language="cn", text=text + "。")


# yingshaoxo_text_to_speech = Yingshaoxo_Text_to_Speech()
# while True:
#     text = input("What you want to say? \n")
#     yingshaoxo_text_to_speech.speak_it_test(text)


speech_recognizer = Speech_Recognizer(language="cn")
yingshaoxo_translator = Yingshaoxo_Translator()
yingshaoxo_text_to_speech = Yingshaoxo_Text_to_Speech()

all_input_text = ""

yingshaoxo_text_to_speech.speak_it("小雅同学初始化完毕")
while True:
    trigger_get_touched = False
    while trigger_get_touched == False:
        trigger_text = speech_recognizer.recognize_following_speech()
        trigger_text = trigger_text.strip()
        if trigger_text in ["你好", "听我说"] or "同学" in trigger_text:
            trigger_get_touched = True
            yingshaoxo_text_to_speech.speak_it("你说")

    while True:
        text = speech_recognizer.recognize_following_speech(timeout_in_seconds=10)
        if text == "":
            # if there has no following speech, we will listen to trigger word than normal sentence
            break
        en_text = yingshaoxo_translator.chinese_to_english(text, sentence_seperation=False)

        print(text)
        print(en_text)

        all_input_text += en_text + "\n" 

        real_input = all_input_text[-8000:].strip()
        response = yingshaoxo_text_generator.search_and_get_following_text_in_a_exact_way(input_text=real_input, quick_mode=False)
        response = decode_response(text=response, chat_context=all_input_text)

        # all_input_text += response + "\n" 

        response = yingshaoxo_translator.english_to_chinese(response, sentence_seperation=False)

        print("\n\n---------\n\n")
        yingshaoxo_text_to_speech.speak_it(response)
        print(response)
        yingshaoxo_text_to_speech.speak_it("我讲完了")
        print("\n\n---------\n\n")