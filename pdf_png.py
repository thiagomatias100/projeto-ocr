from pdf2image import convert_from_path
import cv2
#from PIL import Image, ImageEnhance, ImageOps
import time
import os


#CASO EU DESEJE SALVAR ALGUMAS IMAGEM
#INPUT_FOLDER = "input"
#OUTPUT_FOLDER = "output"

#def in_path(filename):
#   return os.path.join(INPUT_FOLDER, filename)
def pdf_convert_png(pdf_path):
    formato = convert_from_path(pdf_path, dpi=300)
    gray = cv2.imread(formato)
    imagem = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    return imagem 



#IMPLEMENTARIEI PARA MUITAS PAGINAS DEPOIS
#for i , pagina in enumerate(paginas):
#    pagina.save(f"input/diplomamat{i+1}.png","PNG")
#-----------------------------------------------------



#--------------------------------------------------


#PARA ABRIR VARIAS PAGINAS NO FUTURO
# abrir a imagem
#for i,pagina in enumerate(paginas):
    # abrir a imagem
   # img = Image.open(f"input/diplomamat{i+1}.png")
 
   # converter para RGB (importante, senão pode dar erro com PNG ou L)
   # if img.mode in ("RGBA", "P"):
   #     img = img.convert("RGB")
   ## salvar como PDF
   # img.save(f"diplomamat{i+1}ocr.pdf")
    # salvar como PDF


    
     