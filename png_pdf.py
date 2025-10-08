from PIL import Image
from utils import out_file


def img_xerox():
    # abrir a imagem
    img = Image.open(out_file("pb_cpf1.png"))
    # converter para RGB (importante, senão pode dar erro com PNG ou L)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # salvar como PDF
    img.save(out_file("cpf1_pdf.pdf"))


if __name__ == "__main__":
    img_xerox()