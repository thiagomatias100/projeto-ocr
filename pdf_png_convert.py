from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
import time
import os

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

API_URL = "http://200.137.132.64:5001/v1alpha/convert/source"


def in_path(filename):
    return os.path.join(INPUT_FOLDER, filename)



paginas = convert_from_path("lei.pdf", dpi=300)

for i , pagina in enumerate(paginas):
    pagina.save(f"output/pagina_{i+1}.png","PNG")
#-----------------------------------------------------



#--------------------------------------------------



# abrir a imagem
for i,pagina in enumerate(paginas):
    # abrir a imagem
    img = Image.open(f"output/pagina_{i+1}.png")

    # converter para RGB (importante, senão pode dar erro com PNG ou L)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # salvar como PDF
    img.save(f"output/pg_{i+1}_pdf.pdf")
    # salvar como PDF


    
     