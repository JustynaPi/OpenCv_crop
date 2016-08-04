import numpy as np
import cv2

def create_dilation_output():

    img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE) # konwertuje na b&w
    rows, cols = img.shape
    canny = cv2.Canny(img, 50, 240)
    kernel = np.ones((5,5), np.int)
    new_cammy = canny
    dilation = cv2.dilate(new_cammy, kernel, iterations=5)

    #cv2.imshow('Original', img)
    #cv2.imshow('dilated', dilation)
    # cv2.waitKey()
    cv2.imwrite('dilatedOutput.png', dilation)


create_dilation_output()

dilation_output = create_dilation_output()


img = cv2.imread('image.png')
im = cv2.imread('dilatedOutput.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

idx = 0
cnt = contours[1]

for cnt in contours:
    idx +=1
    x,y,w,h = cv2.boundingRect(cnt)
    roi = img[y:y+h, x:x+w]
    cv2.imwrite(str(idx) + '.png', roi)
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),1)

cv2.imshow('im',img)
cv2.waitKey()




