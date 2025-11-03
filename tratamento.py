import cv2
import os

# ---- caminho de entrada/saída ----
in_path  = r"C:\projetos\imgs\entrada.png"   # mude para o seu arquivo
out_dir  = r"C:\projetos\imgs\saida"
os.makedirs(out_dir, exist_ok=True)

# ---- ler imagem ----
img = cv2.imread(in_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Não consegui ler a imagem: {in_path}")

# ---- converter para gray ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- 1) limiar fixo (se você quiser usar 138) ----
_, thresh_fixed = cv2.threshold(gray, 138, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(out_dir, "thresh_138.png"), thresh_fixed)

# ---- 2) Otsu (escolhe o limiar automaticamente) ----
# útil quando a iluminação varia; ignora o valor do limiar passado (0)
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(out_dir, "thresh_otsu.png"), thresh_otsu)

# ---- 3) Adaptativo (bom para documentos com sombras) ----
thresh_adapt = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # ou MEAN_C
    cv2.THRESH_BINARY,
    31,   # tamanho do bloco (ímpar)
    8     # constante subtraída
)
cv2.imwrite(os.path.join(out_dir, "thresh_adapt.png"), thresh_adapt)

print("Arquivos salvos em:", out_dir)


