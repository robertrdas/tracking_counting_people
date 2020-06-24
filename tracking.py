import cv2
import numpy as np
import copy
from Blob import Blob


def read_imput():

	cap = cv2.VideoCapture("imput/768x576.avi")
	frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

	return cap,frame_width,frame_height

def generate_output():

	fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #codificação de saida
	out = cv2.VideoWriter("output/output.avi", fourcc, 5.0, (1280,720))

	return out

if __name__ == '__main__':
	cap,width,higth = read_imput()

	ret, frame_1 = cap.read() #ler o primeiro frame
	ret, frame_2 = cap.read() #ler o segundo frame

	current_frame = copy.deepcopy(frame_1)
	next_frame = copy.deepcopy(frame_2)
	while cap.isOpened() and ret == True:
		
		current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) #converte para nivel de cinza
		next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) #converte para nivel de cinza

		current_frame_gray_blur = cv2.GaussianBlur(current_frame_gray, (5,5), 0)
		next_frame_gray_blur = cv2.GaussianBlur(next_frame_gray, (5,5), 0)

		frame_dif = cv2.absdiff(current_frame_gray_blur, next_frame_gray_blur)

		_, frame_binary = cv2.threshold(frame_dif, 20, 255, cv2.THRESH_BINARY)

		kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		kernel5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		kernel7x7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		kernel9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

		frame_dilate = cv2.dilate(frame_binary,kernel5x5,iterations=2)
		frame_erode = cv2.erode(frame_dilate, kernel5x5, iterations=1)

		frame_erode_copy = copy.deepcopy(frame_erode)

		contours, _ = cv2.findContours(frame_erode_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		coutours_means = 0
		for contour in contours:
			coutours_means = coutours_means + cv2.contourArea(contour)
		coutours_means = coutours_means//len(contours)

		convex_hulls = []
		for contour in contours:
			hull = cv2.convexHull(contour)
			convex_hulls.append(hull)

		blobs = []
		for hull in convex_hulls:
			possibleBlob = Blob(hull)
			possibleBlob.calculate_metrics()

			if possibleBlob.area > 500 and possibleBlob.db_aspect_ratio >=0.2 and possibleBlob.db_aspect_ratio<=1.2 and possibleBlob.w>15 and possibleBlob.h>30 and possibleBlob.db_diagonal_size > 40:
				blobs.append(possibleBlob)
		# for contour in contours:
		# 	(x, y, w, h) = cv2.boundingRect(contour)
		# 	hight_ = abs(y-h)
		# 	widht_ = abs(x-w)
		# 	if cv2.contourArea(contour) < coutours_means*0.68:
		# 		continue
		# 	cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		for blob in blobs:
			cv2.rectangle(current_frame,(blob.x,blob.y),(blob.x+blob.w,blob.y+blob.h), (255, 0, 0), 2)
			cv2.circle(current_frame,(int(blob.center_position_x),int(blob.center_position_y)),3,(0,255,0),-1)

		cv2.putText(current_frame, "Pessoas Detectadas: {}".format(len(blobs)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 255), 3)

		if cv2.waitKey(40) == 27:
			break
		
		cv2.imshow("CURRENT_FRAME", current_frame)
		#cv2.imshow("NEXT_FRAME", next_frame)

		current_frame = next_frame #troca os frames
		ret, next_frame = cap.read() #recupera o segundo frame para continuar a animação

	cv2.destroyAllWindows()
	cap.release()

	
