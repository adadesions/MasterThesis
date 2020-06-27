yesNameList = []
noNameList = []

with(open('data/yes_img_name.txt', 'r')) as _file:
    for line in _file:
        yesNameList.append(line.replace('\n', ''))

print(yesNameList)