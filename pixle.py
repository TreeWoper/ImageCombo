import cv2

img = cv2.imread("data/test/raw_117_0.jpg")

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at:", x, y)

cv2.namedWindow("tile", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("tile", on_click)

while True:
    cv2.imshow("tile", img)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
