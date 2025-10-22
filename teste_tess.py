import cv2
import pytesseract
#import pandas
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

<<<<<<< HEAD
imagem = cv2.imread("cnh.png")
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#imagem = cv2.GaussianBlur(imagem,(3,3),0)
=======
gray = cv2.imread(r"C:\Users\UFMA\Desktop\Nova pasta\cpf.png")
imagem = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
#blur = cv2.GaussianBlur(imagem,(3,3),0)
>>>>>>> 714ad2d8fa4e41c4dc24a3387eafa8e927af2537
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
#imagem = cv2.bilateralFilter(clahe, d=5, sigmaColor=30, sigmaSpace=15)
#reforço de nitidez
sharp = cv2.addWeighted(gray, 1.6, imagem, -0.6, 0)

cv2.imshow( "IMAGEM" ,sharp)
cv2.waitKey(0)


texto = pytesseract.image_to_string(imagem, lang="por")

print(texto)