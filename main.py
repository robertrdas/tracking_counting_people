import numpy as np
import cv2


cap = cv2.VideoCapture("768x576.avi")

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret==False:
		break
	cv2.imshow("frame",frame)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()