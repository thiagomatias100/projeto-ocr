import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pathlib import Path

IMG = "cpf.png"  # <-- seu arquivo

def upscale_if_small(img, min_h=1000):
    h, w = img.shape[:2]
    if h < min_h:
        s = min_h / h
        img = cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_CUBIC)
    return img

def clean_and_binarize(bgr):
    # denoise suave preservando bordas
    den = cv.bilateralFilter(bgr, d=9, sigmaColor=60, sigmaSpace=60)

    # contraste (CLAHE no canal L de LAB)
    lab = cv.cvtColor(den, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    bgr2 = cv.cvtColor(cv.merge([L2, A, B]), cv.COLOR_LAB2BGR)

    # nitidez leve: unsharp
    blur = cv.GaussianBlur(bgr2, (0,0), 1.2)
    sharp = cv.addWeighted(bgr2, 1.2, blur, -0.2, 0)

    # versão para OCR
    gray = cv.cvtColor(sharp, cv.COLOR_BGR2GRAY)
    gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    ocr_bin = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 41, 15)
    return sharp, ocr_bin

def retipar_texto(ocr_ready_bgr, ocr_bin):
    # usa imagem binarizada para melhor detecção
    rgb = cv.cvtColor(ocr_ready_bgr, cv.COLOR_BGR2RGB)
    # forçar OCR com boxes
    config = "--oem 3 --psm 6 -l por"
    data = pytesseract.image_to_data(rgb, config=config, output_type=pytesseract.Output.DICT)

    # cria um "canvas" limpo do mesmo tamanho, cor sólida:
    h, w = ocr_bin.shape[:2]
    canvas = Image.new("RGB", (w, h), (42, 93, 184))  # azul sólido neutro
    draw = ImageDraw.Draw(canvas)

    # fonte genérica (ajuste o caminho/size se quiser)
    # No Windows, pode usar "C:\\Windows\\Fonts\\arial.ttf"
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    # escreve novamente apenas o TEXTO reconhecido, na mesma posição
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i].isdigit() else -1
        if conf < 50 or not txt:
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # leve pad para melhorar legibilidade
        draw.text((x, y), txt, fill=(255,255,255), font=font)

    out = Path(IMG).with_name(Path(IMG).stem + "_retipesado.png")
    canvas.save(out)
    return out

if __name__ == "__main__":
    bgr = cv.imread(IMG, cv.IMREAD_COLOR)
    assert bgr is not None, f"Não achei {IMG}"
    bgr = upscale_if_small(bgr, 1100)
    clean, bin_ = clean_and_binarize(bgr)
    p = Path(IMG)
    cv.imwrite(p.with_name(p.stem+"_clean.png").as_posix(), clean)
    cv.imwrite(p.with_name(p.stem+"_ocr.png").as_posix(), bin_)
    out = retipar_texto(clean, bin_)
    print("Gerados:",
          p.with_name(p.stem+"_clean.png"),
          p.with_name(p.stem+"_ocr.png"),
          out)
