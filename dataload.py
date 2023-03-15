import numpy as np




with open("data/train/neg/0.txt") as f:
    # 获取当前路径

    reviews = f.read()
    labels = "neg"


print(reviews,labels)



# with oepn("data/train/neg")