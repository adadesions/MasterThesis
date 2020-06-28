import cv2
import numpy as np
import pandas as pd


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global idxClass, store, uniPoints, pCount
    global img, imgName, imgCounter
    global yesNameList, noNameList

    sampleClass = param['sampleClass']

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        store[sampleClass[idxClass]+'_x'].append(x)
        store[sampleClass[idxClass]+'_y'].append(y)
        pCount += 1
        print('{} at p{}: ({}, {})'.format(sampleClass[idxClass], pCount, x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if idxClass < len(sampleClass) - 1:
            idxClass += 1
            pCount = 0
            print('Class changed to', sampleClass[idxClass])
        else:
            saveFileAs = 'data/pointSets/points-{}.csv'.format(imgName)
            print('SaveAs {}'.format(saveFileAs))
            df = pd.DataFrame()

            for className in store.keys():
                df[className] = np.array(store[className]).reshape(-1)

            print(df)
            print('Done for {}, Next please!'.format(imgName))

            # Save data to .csv
            df.to_csv(saveFileAs)

            # Clearing state
            idxClass = 0
            pCount = 0
            store = {
                'tumor_x': [],
                'tumor_y': [],
                'normal_x': [],
                'normal_y': [],
                'outer_x': [],
                'outer_y': []
            }

            # Move to the next image
            imgCounter += 1
            imgName = yesNameList[imgCounter]
            img = cv2.imread('data/yes/'+imgName)
            img, uniPoints = plotUniformly(img, offset=10)


def plotUniformly(sourceImg, offset=10):
    uniPoints = []
    img = sourceImg.copy()
    h, w, ch = img.shape

    for col in np.arange(0, h, offset):
        for row in np.arange(0, w, offset):
            uniPoints.append((row, col))
            cv2.circle(img, (row, col), 4, (255, 0, 0), -1)

    return img, uniPoints


if __name__ == '__main__':
    # Setting State
    # Sample Classes
    sampleClass = ['tumor', 'normal', 'outer']
    idxClass = 0
    imgCounter = 0
    pCount = 0

    # Declare State Store
    store = {
        'tumor_x': [],
        'tumor_y': [],
        'normal_x': [],
        'normal_y': [],
        'outer_x': [],
        'outer_y': []
    }

    yesNameList = []
    noNameList = []

    with(open('data/yes_img_name.txt', 'r')) as _file:
        for line in _file:
            yesNameList.append(line.replace('\n', ''))

    # Create a black image, a window and bind the function to window
    windowName = 'Training Set'
    imgName = yesNameList[imgCounter]
    img = cv2.imread('data/yes/'+imgName)
    img, uniPoints = plotUniformly(img, offset=10)
    mouseParam = {
        'sampleClass': sampleClass,
    }
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, draw_circle, mouseParam)

    while(1):
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
