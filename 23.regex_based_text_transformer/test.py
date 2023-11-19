import re

regex = r"Are you(?P<y0>.*?)\?"
text = "Are you smart?"

match = re.search(regex, text)
if match:
    print(match.group('y0'))
else:
    print("No match found")
