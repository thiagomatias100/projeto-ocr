import os
import sys
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# ====== CONFIGURAÇÕES RÁPIDAS ======
IMAGE_PATH = r"lei.png"  # <<< troque para sua imagem
# Windows: ajuste para o seu caminho do executável do Tesseract:
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "por+eng"   # português + inglês
EASYOCR_LANGS = ['pt', 'en']
USE_GPU = False         # mude para True se tiver CUDA
CONF_THRESH = 0.35      # confiança mínima para exibir (0–1)
TARGET_TEXT_HEIGHT = 24 # px mínimos de altura de texto (ajuda os dois)

# ====== OPÇÃO: LEVE PRÉ-PROCESSAMENTO (ajuda, mas é opcional para EasyOCR) ======
def light_preprocess(bgr):
    # cinza + CLAHE leve + resize para garantir altura mínima de texto
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    # estima altura média dos caracteres pelo gradiente (heurístico bem simples)
    # se a imagem estiver "pequena", ampliamos
    h, w = clahe.shape
    # alvo: que a menor dimensão tenha ~1000 px (heurística simples)
    scale = 1.0
    min_dim_target = 1000
    if min(h, w) < min_dim_target:
        scale = min_dim_target / float(min(h, w))
    if scale != 1.0:
        clahe = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return clahe

# ====== DESENHO DE BBOXs ======
def draw_boxes_bgr(img_bgr, boxes, color=(0,255,0), thickness=2, with_text=True):
    out = img_bgr.copy()
    for b in boxes:
        (x1, y1, x2, y2), text, conf = b
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
        if with_text and text:
            label = f"{text} ({conf:.2f})"
            # legenda acima da caixa
            cv2.putText(out, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1, cv2.LINE_AA)
    return out

# ====== EASYOCR ======
def run_easyocr(img_bgr):
    try:
        import easyocr
    except ImportError:
        print("EasyOCR não instalado. Rode: pip install easyocr")
        return []

    reader = easyocr.Reader(EASYOCR_LANGS, gpu=USE_GPU)
    # EasyOCR aceita BGR/GRAYSCALE/array; vamos passar a imagem em tons de cinza melhorada
    pre = light_preprocess(img_bgr)

    results = reader.readtext(pre, detail=1)  # [ [box, text, conf], ... ]
    boxes = []
    # cada box vem como 4 pontos (quadrilátero); faremos o bbox retangular mínimo
    for box, text, conf in results:
        # conf do easyocr já vem 0..1 (na maioria das versões)
        if conf is None:
            continue
        if conf < CONF_THRESH:
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append(((x1, y1, x2, y2), text, float(conf)))
    return boxes

# ====== TESSERACT ======
def run_tesseract(img_bgr):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD if os.path.exists(TESSERACT_CMD) else pytesseract.pytesseract.tesseract_cmd
    pre = light_preprocess(img_bgr)

    # Use image_to_data para obter bboxes por palavra
    # OEM 3 (default LSTM) e PSM 6 (assume bloco com uma única coluna de texto)
    config = f'--oem 3 --psm 6 -l {TESS_LANG}'
    data = pytesseract.image_to_data(pre, config=config, output_type=pytesseract.Output.DICT)

    boxes = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = data['conf'][i]
        if text == "" or text.isspace():
            continue
        try:
            conf_val = float(conf)
        except:
            continue
        # Tesseract costuma dar confiança 0..100; normalizamos para 0..1
        conf_norm = max(0.0, min(1.0, conf_val / 100.0))
        if conf_norm < CONF_THRESH:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append(((x, y, x+w, y+h), text, conf_norm))
    return boxes

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Imagem não encontrada: {IMAGE_PATH}")
        sys.exit(1)

    # lê em BGR
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Falha ao carregar a imagem. Verifique o caminho e extensão.")
        sys.exit(1)

    # roda ambos
    easy_boxes = run_easyocr(img_bgr)
    tess_boxes = run_tesseract(img_bgr)

    # gera visuais
    easy_viz = draw_boxes_bgr(img_bgr, easy_boxes, color=(0,255,0))
    tess_viz = draw_boxes_bgr(img_bgr, tess_boxes, color=(255,0,0))

    # junta lado a lado
    h1, w1 = easy_viz.shape[:2]
    h2, w2 = tess_viz.shape[:2]
    H = max(h1, h2)
    # pad para alturas iguais
    def pad_to_h(img, H):
        h, w = img.shape[:2]
        if h == H: return img
        pad = np.zeros((H-h, w, 3), dtype=img.dtype)
        return np.vstack([img, pad])
    easy_pad = pad_to_h(easy_viz, H)
    tess_pad = pad_to_h(tess_viz, H)
    side_by_side = np.hstack([easy_pad, tess_pad])

    # salva e mostra
    out_path = "ocr_compare_easyocr_tesseract.png"
    cv2.imwrite(out_path, side_by_side)
    print(f"Resultado salvo em: {out_path}")
    print(f"EasyOCR: {len(easy_boxes)} boxes | Tesseract: {len(tess_boxes)} boxes")

    # exibe com matplotlib (BGR->RGB)
    plt.figure(figsize=(14, 8))
    plt.imshow(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Esquerda: EasyOCR (verde) • Direita: Tesseract (vermelho)")
    plt.show()

if __name__ == "__main__":
    main()
