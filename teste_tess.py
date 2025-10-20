import cv2
import pytesseract
#import pandas
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as pltS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#gray = cv2.imread(r"C:\Users\UFMA\Desktop\Nova pasta\cpf.png")
#imagem = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
#blur = cv2.GaussianBlur(imagem,(3,3),0)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
#imagem = cv2.bilateralFilter(clahe, d=5, sigmaColor=30, sigmaSpace=15)
#reforço de nitidez
#sharp = cv2.addWeighted(gray, 1.6, imagem, -0.6, 0)

#cv2.imshow( "IMAGEM" ,sharp)
#cv2.waitKey(0)
img = cv2.imread(r"C:\Users\UFMA\Desktop\Nova pasta\cnh.png")
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
texto = pytesseract.image_to_data(img,lang='por', config='--psm 3')


img2 = Image.open(r"C:\Users\UFMA\Desktop\Nova pasta\cnh.png")
plt.imshow(img2)
plt.show()
print(texto )

