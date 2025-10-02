from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

def enhance_for_ocr_pillow(img: Image.Image, upscale_to_dpi=300) -> Image.Image:
    # 1) garantir resolução mínima (ótimo p/ OCR)
    if upscale_to_dpi:
        # assume que a imagem veio de PDF 72dpi; faz upscale proporcional ≈ 4x
        scale = upscale_to_dpi / 72
        w, h = img.size
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # 2) escala de cinza + equalização leve
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)

    # 3) correções globais
    img = ImageEnhance.Brightness(img).enhance(1.1)  # +10% brilho
    img = ImageEnhance.Contrast(img).enhance(1.4)    # +40% contraste

    # 4) redução de ruído leve + realce de nitidez
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))

    # 5) binarização simples (rápida; troque por Otsu/Adaptativa com OpenCV se precisar)
    arr = np.array(img)
    thresh = 180
    arr = (arr > thresh) * 255
    return Image.fromarray(arr.astype("uint8"))


# 1) abrir a imagem original
img = Image.open("pagina_4.png")

# 2) aplicar a função de aprimoramento
proc = enhance_for_ocr_pillow(img)

# 3) salvar o resultado
proc.save("pagina__proc.png")
