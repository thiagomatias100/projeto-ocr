# Demo: realce de contorno e reforço de áreas brancas (textos) usando PIL + NumPy + Matplotlib
# - Se você salvar uma imagem como /mnt/data/input.png, o script vai usá-la.
# - Caso contrário, ele gera uma imagem sintética com texto pálido num fundo cinza.
#
# Saídas:
#   - Três figuras (Original, Realce de contorno, Branco intensificado)
#   - Arquivo salvo: /mnt/data/realce_texto.png

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import matplotlib.pyplot as plt

def generate_synthetic(w=1000, h=350):
    bg = Image.new("L", (w, h), 180)  # fundo cinza claro
    draw = ImageDraw.Draw(bg)
    # fonte padrão (fallback)
    text = "Documento UFMA - Prova de Conceitos em OCR"
    text2 = "CPF: 123.456.789-00   RG: 0000000-0   NOME: THIAGO"
    # posiciona com leve variação para parecer 'escaneado'
    draw.text((40, 60), text, fill=235)      # texto branco pálido
    draw.text((40, 140), text2, fill=230)
    # linhas mais finas (como carimbo/apagado)
    draw.line((40, 210, 960, 210), fill=220, width=1)
    draw.text((40, 240), "Endereço: Av. dos Portugueses, 1966 - São Luís/MA", fill=225)
    # ruído suave
    arr = np.array(bg, dtype=np.float32)
    noise = np.random.normal(0, 4, arr.shape)  # ruído baixo
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    # desfocar levemente (simula foco ruim)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img

def percentile_stretch(img, p_low=2, p_high=98):
    a = np.array(img, dtype=np.float32)
    lo = np.percentile(a, p_low)
    hi = np.percentile(a, p_high)
    if hi <= lo:
        return img
    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0, 1)
    return Image.fromarray((a*255).astype(np.uint8), mode="L")

def unsharp_mask(img, blur_radius=1.5, amount=1.2):
    # Unsharp clássico: sharp = original + amount*(original - blur)
    blur = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    a = np.array(img, dtype=np.float32)
    b = np.array(blur, dtype=np.float32)
    sharp = a + amount*(a - b)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp, mode="L")

def whiten_intensify(img, thr=0.85, factor=1.35, maxfilter=3):
    # img esperado em tons de cinza 0..255
    a = np.array(img, dtype=np.float32) / 255.0
    mask = (a >= thr).astype(np.float32)
    # reforça regiões brancas mantendo contornos já realçados
    enhanced = a*(1 - mask) + np.clip(a*factor, 0, 1)*mask
    out = Image.fromarray((enhanced*255).astype(np.uint8), mode="L")
    # leve dilatação para engrossar letras finas
    out = out.filter(ImageFilter.MaxFilter(size=maxfilter))
    return out

# 1) Carregar ou criar imagem
in_path = "cpf.png"
if os.path.exists(in_path):
    img0 = Image.open(in_path).convert("L")
else:
    img0 = generate_synthetic()

# 2) Correção simples de contraste por percentis
img1 = percentile_stretch(img0, p_low=2, p_high=98)

# 3) Realce de contorno por unsharp masking
img2 = unsharp_mask(img1, blur_radius=1.2, amount=1.4)

# 4) Reforçar áreas brancas (onde costuma estar o texto claro/acinzentado)
img3 = whiten_intensify(img2, thr=0.82, factor=1.45, maxfilter=3)

# 5) Mostrar resultados (cada figura isolada, sem subplots)
plt.figure()
plt.title("Original (cinza)")
plt.imshow(img0, cmap="gray")
plt.axis("off")

plt.figure()
plt.title("Após realce de contorno (Unsharp)")
plt.imshow(img2, cmap="gray")
plt.axis("off")

plt.figure()
plt.title("Branco intensificado (texto realçado)")
plt.imshow(img3, cmap="gray")
plt.axis("off")

# 6) Salvar saída final
out_path = "cpf.png"
img3.save(out_path)

out_path
