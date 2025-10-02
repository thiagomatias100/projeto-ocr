from PIL import Image, ImageEnhance, ImageOps

def xerox_effect(img):
    # converter para tons de cinza
    img = img.convert("L")
    
    # equalizar o histograma → melhora contraste
    img = ImageOps.equalize(img)
    
    # aumentar contraste
    img = ImageEnhance.Contrast(img).enhance(5)
    
    # reduzir brilho para dar aquele “cinza xerox”
    img = ImageEnhance.Brightness(img).enhance(1.8)
    
    return img
img = Image.open("pagina_1.png")
xerox_img = xerox_effect(img)
xerox_img.save("pagina_1_xerox.png")
