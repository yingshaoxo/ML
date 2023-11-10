#pip3 install "git+https://github.com/yingshaoxo/auto_everything.git@dev"
#pip install selenium
#pip install webdriver-manager

from auto_everything.web import Selenium
my_selenium = Selenium("https://heypi.com/talk", headless=False, use_firefox=False, user_data_dir="C:/Users/yingshaoxo/AppData/Local/Google/Chrome/User Data")
d = my_selenium.driver

from common_functions import *

# All functions and variables in code block will be in memory forever.
# Every code block has to get excuted.

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