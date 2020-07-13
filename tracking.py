import cv2
import numpy as np
import copy
from Blob import Blob


save_images = False #TRUE: salvará uma imagem para cada etapa do metodo FALSE: irá salvar apenas o video de saida
threshold_people = 5 #limite de pssoas , utilizado no teste


#realiza a leitura do video de entrada
def read_imput():

	cap = cv2.VideoCapture("imput/768x576.avi")
	frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

	return cap,frame_width,frame_height

#gera o arquivo do video que é usado para salvar o video de saida
def generate_output():

	fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #codificação de saida
	out = cv2.VideoWriter("output/output.avi", fourcc, 5.0, (1280,720))

	return out

#utilizada para salvar as imagens de etapas do metodo
#caso a flag save_image esteja como true as imagens do passso a passo serão salvas
def save_image_step(name_image,frame_count,image):
	path = "images/"+str(name_image)+"_"+str(frame_count)+".jpg"
	if save_images:
		cv2.imwrite(path,image)

#implementação do melhoramento de contraste DECOMPOSIÇÂO INTRINSECA
def intrinsic_decomposition(image):
	imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	V = imgHSV[:,:,2]
	rows, cols = V.shape
	La = np.zeros((rows, cols))
	V = np.float32(V)
	for i in range(rows):
		for j in range(cols):
			La[i,j] = 255*((V[i,j]/255)**(1/2.2))
	V = np.uint8(V) 
	La = np.uint8(La)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(La)
	newHSV = cv2.merge([imgHSV[:,:,0], imgHSV[:,:,1], cl1])
	result = cv2.cvtColor(newHSV, cv2.COLOR_HSV2BGR)
	return result

#conjunto de funções para o filtro gaussiano
def gaussXY(x,y,sig):
	fator = (1/(2*mt.pi*(sig**2)))

	g = fator*mt.exp( -((x**2 + y**2)/(2*(sig**2))) )

	return g

def mascara_gauss(n, sig):
	resultado = np.zeros((n,n))

	a = n//2
	b = n//2

	for x in range(-a,a):
		for y in range(-b,b):
			resultado[x+a+1,y+b+1] = gaussXY(x,y,sig) 
	
				
	resultado = resultado/np.sum(resultado)
	return list(resultado.flatten())

#implementação propria do filtro gaussiano
def gaussian_blur(img,n,sig):
	valor = 0
	aresta  = n//2
	arestas =[]

	if n == 3:
		mascara = list(np.array([1,2,1,2,4,2,1,2,1])/16) #usado para mascara 3x3
	elif n==5:
		mascara = list(np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1])/273) #usado para mascar 5x5
	else:
		return None
		
	linhas,colunas = img.shape
	imagem_resultado = np.zeros((linhas,colunas),np.uint8)

	for i in range(aresta,linhas-aresta):
		for j in range(aresta,colunas-aresta):

			for x in range(n):
				for y in range(n):
					arestas.append(img[i-aresta+x,j-aresta+y])					
			
			for k in range(len(arestas)):
				valor += arestas[k] * mascara[k]

			resultado = round((valor))
			
			if resultado < 0 : 
				resultado = 0
			if resultado > 255: 
				resultado = 255

			imagem_resultado[i,j] = resultado

			arestas.clear()
			valor = 0
	return imagem_resultado


#função principal
if __name__ == '__main__':

	frame_count = 1
	cap,width,higth = read_imput() #ler o video de entrada
	out = generate_output() #gera o video de saida
	ret, frame_1 = cap.read() #ler o primeiro frame
	ret, frame_2 = cap.read() #ler o segundo frame

	current_frame = copy.deepcopy(frame_1) #realiza a copia da imagem 1
	next_frame = copy.deepcopy(frame_2) #realiza a copia da imagem 2

	#processar cada frame do video
	while cap.isOpened() and ret == True:

		#salva imagem original
		save_image_step("original",frame_count,next_frame)

		current_frame_decomposition = intrinsic_decomposition(current_frame)
		next_frame_decomposition = intrinsic_decomposition(next_frame)

		#salva imagem resultante da decomposição intrinseca
		save_image_step("decomposition",frame_count,next_frame_decomposition)

		current_frame_gray = cv2.cvtColor(current_frame_decomposition, cv2.COLOR_BGR2GRAY) #converte para nivel de cinza
		next_frame_gray = cv2.cvtColor(next_frame_decomposition, cv2.COLOR_BGR2GRAY) #converte para nivel de cinza

		#salva imagem em nivel de cinza
		save_image_step("gray",frame_count,next_frame_gray)

		current_frame_gray_blur = cv2.GaussianBlur(current_frame_gray, (5,5), 0)
		#current_frame_gray_blur = gaussian_blur(current_frame_gray, 5, 1)
		next_frame_gray_blur = cv2.GaussianBlur(next_frame_gray, (5,5), 0)
		#next_frame_gray_blur = gaussian_blur(next_frame_gray, 5, 1)

		#salva a imagem suavizada
		save_image_step("blur",frame_count,next_frame_gray_blur)

		frame_dif = cv2.absdiff(current_frame_gray_blur, next_frame_gray_blur) #realiza a subtração de imagens

		#salva a imagem resultado da diferença
		save_image_step("diff",frame_count,frame_dif)

		_, frame_binary = cv2.threshold(frame_dif, 20, 255, cv2.THRESH_BINARY)

		#salva a imagem binaria
		save_image_step("binary",frame_count,frame_binary)

		#kernels utilizados para testes das operações morfologicas
		kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		kernel5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		kernel7x7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		kernel9x9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

		frame_dilate = cv2.dilate(frame_binary,kernel5x5,iterations=2)

		#salva a imagem após a dilatação
		save_image_step("dilate",frame_count,frame_dilate)

		frame_erode = cv2.erode(frame_dilate, kernel5x5, iterations=3)

		#salva a imagem após a erosão
		save_image_step("erode",frame_count,frame_erode)

		frame_erode_copy = copy.deepcopy(frame_erode)

		#realiza a detecção de contornos
		contours, _ = cv2.findContours(frame_erode_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		#dimessões da imagem para salvar o resultado do fechoconvexo para teste
		linha,coluna = current_frame_gray.shape

		#calcula os fechos convexos para cada contorno
		convex_hulls = []
		for contour in contours:
			hull = cv2.convexHull(contour)
			convex_hulls.append(hull)

		#cria a imagem para o fecho convexo
		imagem_cnt = np.zeros((linha,coluna),np.uint8)

		#desenha os "contornos" do fecho convexo na imagem
		cv2.drawContours(imagem_cnt, convex_hulls, -1, (255,255,255), 3)

		#salva a imagem do fecho convexo
		save_image_step("convex_hull",frame_count,imagem_cnt)

		#REFINAMENTO DAS HIPOTESES
		#para cada possivel hipotese é filtrada
		blobs = []
		for hull in convex_hulls:
			possibleBlob = Blob(hull)
			possibleBlob.calculate_metrics()

			#filtra as hipoteses considerando as propriedades calculadas
			if possibleBlob.area > 500 and possibleBlob.db_aspect_ratio >=0.2 and possibleBlob.db_aspect_ratio<=1.2 and possibleBlob.w>15 and possibleBlob.h>30 and possibleBlob.db_diagonal_size > 40:
				blobs.append(possibleBlob)
		# for contour in contours:
		# 	(x, y, w, h) = cv2.boundingRect(contour)
		# 	hight_ = abs(y-h)
		# 	widht_ = abs(x-w)
		# 	if cv2.contourArea(contour) < coutours_means*0.68:
		# 		continue
		# 	cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		#desenha os boundbox dos obejtos encontrados
		for blob in blobs:
			if len(blobs) > threshold_people:
				cv2.rectangle(current_frame,(blob.x,blob.y),(blob.x+blob.w,blob.y+blob.h), (0, 0, 255), 2)
				cv2.circle(current_frame,(int(blob.center_position_x),int(blob.center_position_y)),3,(0,0,255),-1)
			else:
				cv2.rectangle(current_frame,(blob.x,blob.y),(blob.x+blob.w,blob.y+blob.h), (0, 255, 0), 2)
				cv2.circle(current_frame,(int(blob.center_position_x),int(blob.center_position_y)),3,(0,255,0),-1)

		#inseri o texto com a quantidade de pessoas
		cv2.putText(current_frame, "Pessoas Detectadas: {}".format(len(blobs)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 0), 3)

		if cv2.waitKey(40) == 27:
			break
		
		cv2.imshow("CURRENT_FRAME", current_frame)

		#salva a imagem final
		save_image_step("finaly",frame_count,current_frame)
		#cv2.imshow("NEXT_FRAME", next_frame)

		#realiza o redmensionamento
		image_video = cv2.resize(current_frame, (1280,720))
		#escreve a imagem no video de saida
		out.write(image_video)

		current_frame = next_frame #proximo frame vira o atual
		ret, next_frame = cap.read() #recupera o proximo frame
		frame_count = frame_count + 1

	cv2.destroyAllWindows()
	cap.release()

	
