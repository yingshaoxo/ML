import os

num = 0
for root, dirs, files in os.walk(".", topdown = False):
    for name in files:
        if "dataset" in root:
            os.rename(os.path.join(root, name), os.path.join(root,  str(num) + "." +  name.split('.')[-1]))
            num += 1
