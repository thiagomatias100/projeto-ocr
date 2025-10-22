import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#pil_img = Image.open(r"C:\Users\UFMA\Desktop\Nova pasta\diplomamat1.png")
img = cv2.imread(r"C:\Users\UFMA\Desktop\Nova pasta\cpf.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

pil_img = Image.fromarray(img)

resultado = pytesseract.image_to_data(pil_img, lang='por', output_type=Output.DICT)

def text_box(resultado, img):
    n = len(resultado['text'])
    for i in range(n):
        txt = (resultado['text'][i] or "").strip()
        if not txt:
            continue
        x = int(resultado['left'][i])
        y = int(resultado['top'][i])
        w = int(resultado['width'][i])
        h = int(resultado['height'][i])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, txt, (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# converte PIL -> OpenCV
img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
text_box(resultado, img_cv)

cv2.imwrite("com_bboxes.png", img_cv)
print("Imagem salva com boxes em 'com_bboxes.png'")
print(pytesseract.image_to_string(pil_img, lang="por", config="--psm 1"))
