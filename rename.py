import os

path = "./data/ShanghaiA/train/"
files = os.listdir(path)
train = []
for f in files:
    print(f)

    start = f.split(".")[0]
    end = f.split(".")[1]
    if(end == "jpg"):
        train.append(f)
    #elif end == "mat":
        #num = start.split("_")[-1]
        #os.rename(path + f, path + "IMG_" + num + "_ann.mat")
    #print(end)

with open('train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

path = "./data/ShanghaiA/test/"
files = os.listdir(path)
train = []
for f in files:
    print(f)

    start = f.split(".")[0]
    end = f.split(".")[1]
    if(end == "jpg"):
        train.append(f)
    elif end == "mat":
        num = start.split("_")[-1]
        os.rename(path + f, path + "IMG_" + num + "_ann.mat")
    #print(end)

with open('val.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)