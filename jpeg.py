import cv2
img = cv2.imread("train.png")
# cv2.imshow("windows",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('train_50.jpeg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])