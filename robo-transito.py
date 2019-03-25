import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_inRange(img, mask):
    imask = mask > 0
    sliced = np.zeros_like(img, np.uint8)
    sliced[imask] = img[imask]
    plt.subplot(211)
    plt.imshow(sliced)
    plt.subplot(212)
    plt.imshow(img)
    plt.show()


imagem = "teste9.jpg"

img = cv2.imread(imagem)

img = cv2.medianBlur(img, 3)
_, img = cv2.threshold(img,150,255,cv2.ADAPTIVE_THRESH_MEAN_C)

plt.imshow(img)
plt.show()


# ([99, 214, 104]) rgb
minGreen = np.array([0, 200, 0])
# ([186, 237, 189]) rgb
maxGreen = np.array([100, 255, 100])


# azul cyan
minOrange = np.array([0, 100, 100])
maxOrange = np.array([100, 255, 255])

# blue
minRed = np.array([0, 0, 0])
maxRed = np.array([100, 100, 255])

# white -> inv
minWine = np.array([180, 180, 180])
maxWine = np.array([255, 255, 255])


onlyGreen = cv2.inRange(img, minGreen, maxGreen)
onlyOrange = cv2.inRange(img, minOrange, maxOrange)
onlyRed = cv2.inRange(img, minRed, maxRed)

_, im = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
onlyWine = cv2.inRange(im, minWine, maxWine)
plt.imshow(im)
plt.show()

pixelsGreen = cv2.countNonZero(onlyGreen)
pixelsOrange = cv2.countNonZero(onlyOrange)
pixelsRed = cv2.countNonZero(onlyRed)
pixelsWine = cv2.countNonZero(onlyWine)

pixelsTotal = img.size

mostrar_inRange(img, onlyGreen)
mostrar_inRange(img, onlyOrange)
mostrar_inRange(img, onlyRed)
mostrar_inRange(im, onlyWine)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('% Total da Imagem: ')
print('% Verde: ', (pixelsGreen/pixelsTotal)*100)
print('% Laranja: ', (pixelsOrange/pixelsTotal)*100)
print('% Vermelho: ', (pixelsRed/pixelsTotal)*100)
print('% Vinho: ', (pixelsWine/pixelsTotal)*100)

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
total = pixelsGreen + pixelsOrange + pixelsRed + pixelsWine
print('% Total das Rotas: ')
print('% Verde: ', (pixelsGreen/total)*100)
print('% Laranja: ', (pixelsOrange/total)*100)
print('% Vermelho: ', (pixelsRed/total)*100)
print('% Vinho: ', (pixelsWine/total)*100)

# TODO melhorar a precisão da cor Vinho/Preto/Branco -> reconhecento letras

img = cv2.imread(imagem, 0)
_, img = cv2.threshold(img, 110, 255, cv2.AGAST_FEATURE_DETECTOR_THRESHOLD)
img = cv2.medianBlur(img, 3) # cv2.bilateralFilter(img,20,150,150) # cv2.GaussianBlur(img, (3, 3), 3) #
plt.imshow(img)
plt.show()

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5, maxRadius=30) # Your code

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('% Total de Marcadores: ')
print('% Vias Bloqueadas: ')
print('% Acidentes: ')
print('% Trabalhadores: ')

if circles is not None:
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), 10, (0, 255, 255), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)
    plt.imshow(img)
    plt.show()
    print('% Total: ', len(circles[0, :]))
else:
    print('% Total: 0')



'''
import numpy as np
import cv2

im = cv2.imread('teste3.jpg')
im = cv2.medianBlur(im, 3)

_, im = cv2.threshold(im,150,255,cv2.ADAPTIVE_THRESH_MEAN_C)
fator_branco = np.mean(im / 255)
print(str(fator_branco * 100) + "%")
cv2.imshow("Resultado", im)
cv2.waitKey()
'''

