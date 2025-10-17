import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

imagem = cv2.imread("cnh.png")
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#imagem = cv2.GaussianBlur(imagem,(3,3),0)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)


texto = pytesseract.image_to_string(imagem, lang="por")

print(texto)