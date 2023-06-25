#pip3 install "git+https://github.com/yingshaoxo/auto_everything.git@dev"
#pip install selenium
#pip install webdriver-manager

from auto_everything.web import Selenium
from auto_everything.disk import Disk
from auto_everything.io import IO
import text_to_voice

disk = Disk()
io_ = IO()
my_selenium = Selenium("https://heypi.com/talk", headless=False, use_firefox=False, user_data_dir="C:/Users/yingshaoxo/AppData/Local/Google/Chrome/User Data")
d = my_selenium.driver

output_txt_file = "./dataset.txt"
if not disk.exists(output_txt_file):
    io_.write(output_txt_file, "")


# All functions and variables in code block will be in memory forever.
# Every code block has to get excuted.

def general_text_wrapper(who_said: str, language: str, content: str):
    text = f"-{who_said}-:\n"
    content = f"""
```{language}
{content.strip()}
```
    """.strip()
    text += content
    text += "\n\n__**__**__yingshaoxo_is_the_top_one__**__**__\n\n"
    return text

def handle_pi_ai_text(text: str):
    text = text.strip()
    print()
    print(text)
    print()
    print()
    text_to_voice.say_somthing(text)
    if (text == ""):
        return
    with open(output_txt_file, "a", encoding="utf-8", errors="ignore") as f:
        f.write(general_text_wrapper(who_said="pi_ai", language="text", content=text))

def handle_yingshaoxo_ai_text(text: str):
    text = text.strip()
    if (text == ""):
        return
    with open(output_txt_file, "a", encoding="utf-8", errors="ignore") as f:
        text_template = f"""
print('''
{text}
'''.strip())
        """
        f.write(general_text_wrapper(who_said="yingshaoxo", language="python", content=text_template))

last_request = ""
while True:
    xpath = "//div[span and contains(@class, 'whitespace-pre-wrap')]"
    elements = my_selenium.wait_until_elements_exists(xpath)
    if len(elements) == 0:
        exit()
    else:
        text = ""
        for element in elements:
            text_part = d.execute_script('return arguments[0].innerText;', element)
            #text_part = element.text
            text += text_part + "\n"
        # text = text.split("By messaging Pi, you are agreeing to our Terms")[0]
        # if (last_request != ""):
        #     text = text.split(last_request)[1]
        if last_request != "":
            handle_pi_ai_text(text)

    new_request = input("What you want to say?\n")

    elements = my_selenium.wait_until_elements_exists("//textarea[contains(@class, 'overflow-y-hidden')]")
    if len(elements) == 0:
        exit()
    else:
        elements[0].click()
        elements[0].send_keys(new_request)
        elements[0].send_keys("\n")
        last_request = new_request
        handle_yingshaoxo_ai_text(last_request)
        print()
        print()
        print()
        my_selenium.sleep(10)