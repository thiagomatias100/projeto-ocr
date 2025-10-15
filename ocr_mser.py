import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------- 1) Pré-processamento ----------
def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    # leve denoise, preservando arestas
    blur = cv2.bilateralFilter(clahe, d=5, sigmaColor=30, sigmaSpace=15)
    # reforço de nitidez
    sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
    return sharp

# --------- 2) MSER mais restrito ----------
def mser_regions(gray, delta=5, min_area=60, max_area_ratio=0.20):
    h, w = gray.shape
    max_area = int(h*w*max_area_ratio)
    # Use parâmetros adicionais do MSER p/ reduzir blobs “fofos”
    mser = cv2.MSER_create(
        _delta=delta,
        _min_area=min_area,
        _max_area=max_area,
        _max_variation=0.25,    # mais baixo = mais rígido
        _min_diversity=0.2,     # evita regiões muito parecidas
        _max_evolution=200,
        _area_threshold=1.01,
        _min_margin=0.003,
        _edge_blur_size=5
    )
    regions, _ = mser.detectRegions(gray)
    return regions

# --------- 3) Recursos "sensíveis ao traço" ----------
def _box_features(gray, box):
    x0,y0,x1,y1 = box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0: 
        return None

    # (a) edges/area (texto tem boa densidade de borda)
    edges = cv2.Canny(crop, 50, 150, L2gradient=True)
    edge_density = edges.mean() / 255.0

    # (b) binarização + fill ratio (quanto “preto” há)
    _, bin_ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # assumir texto escuro: inverter p/ foreground=1
    fg = (255 - bin_) // 255
    fill_ratio = float(fg.mean())  # ~0.05–0.6 é típico

    # (c) stroke width (aprox.): distance transform no FG
    dt = cv2.distanceTransform((fg*255).astype(np.uint8), cv2.DIST_L2, 3)
    sw = dt[fg.astype(bool)] * 2.0  # aprox largura do traço
    if sw.size < 5:
        sw_cv = 999.0
        sw_mean = 999.0
    else:
        sw_mean = float(sw.mean())
        sw_std  = float(sw.std())
        sw_cv   = sw_std / (sw_mean + 1e-6)

    # (d) razão W/H e “compactação” do conteúdo
    h, w = crop.shape
    aspect = w / max(h,1)
    # razão de preenchimento morfológico (fecha pequenos buracos)
    closed = cv2.morphologyEx((fg*255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    compactness = (closed>0).mean()

    return dict(edge_density=edge_density, fill_ratio=fill_ratio,
                sw_cv=sw_cv, sw_mean=sw_mean, aspect=aspect,
                height=h, width=w, compactness=compactness)

def boxes_from_regions(regions, img_shape):
    H, W = img_shape
    boxes = []
    for pts in regions:
        x,y,w,h = cv2.boundingRect(pts)
        area = w*h
        # 1) cortes brutos por tamanho
        if area < 80 or area > int(W*H*0.20):
            continue
        # 2) cortes por aspecto e altura relativa
        aspect = w / max(h,1)
        if aspect < 0.15 or aspect > 8.0:
            continue
        if h < max(10, H*0.01) or h > H*0.20:
            continue
        # padding leve
        pad = 2
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        box = (x0,y0,x1,y1)

        feats = _box_features(cv2.cvtColor(cv2.cvtColor(np.zeros((H,W,3), dtype=np.uint8), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), box)
        # ^ linha acima foi só placeholder – vamos calcular com o 'gray' certo na função de chamador.
        boxes.append(box)
    return boxes

# >>> Ajuste: calcular features usando o gray correto
def scored_boxes_from_regions(regions, gray):
    H, W = gray.shape
    raw = []
    for pts in regions:
        x,y,w,h = cv2.boundingRect(pts)
        area = w*h
        if area < 80 or area > int(W*H*0.20):
            continue
        aspect = w / max(h,1)
        if aspect < 0.15 or aspect > 8.0:
            continue
        if h < max(10, H*0.01) or h > H*0.20:
            continue
        pad = 2
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        box = (x0,y0,x1,y1)
        feats = _box_features(gray, box)
        if not feats:
            continue

        # filtros por traço:
        if not (0.03 <= feats["edge_density"] <= 0.45):
            continue
        if not (0.08 <= feats["fill_ratio"] <= 0.65):
            continue
        if feats["sw_mean"] > 12:              # traço muito grosso → ruído/figura
            continue
        if feats["sw_cv"] > 0.6:               # variação alta do traço → ruído
            continue
        if feats["compactness"] < 0.2:         # muito “oco”
            continue

        # score simples: favorece boa densidade de borda e preenchimento moderado
        score = (feats["edge_density"]*1.0) + (0.6*feats["fill_ratio"]) + (0.3*(1.0 - min(feats["sw_cv"],1.0)))
        raw.append((box, score))
    return raw

# --------- 4) NMS orientado por score ----------
def suppress_overlaps_scored(scored_boxes, iou_thresh=0.3):
    if not scored_boxes: 
        return []
    rects = np.array([b for b,_ in scored_boxes])
    scores = np.array([s for _,s in scored_boxes])
    x1 = rects[:,0]; y1 = rects[:,1]; x2 = rects[:,2]; y2 = rects[:,3]
    areas = (x2-x1+1)*(y2-y1+1)

    order = scores.argsort()[::-1]  # do maior score para o menor
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds+1]
    return [tuple(rects[i]) for i in keep]

# --------- 5) Agrupar em PALAVRAS (não só linhas) ----------
def group_words(boxes, gap_factor=0.5, y_tol=12):
    """
    gap_factor: múltiplo da largura média da caixa na linha para separar palavras
    """
    if not boxes: return []
    # ordenar por y, depois x
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    # formar linhas por y
    lines = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) <= y_tol:
            current.append(b)
        else:
            lines.append(sorted(current, key=lambda r: r[0]))
            current = [b]
    lines.append(sorted(current, key=lambda r: r[0]))

    # dentro da linha, agrupar por gaps em palavras
    merged_words = []
    for line in lines:
        widths = [bx[2]-bx[0] for bx in line]
        avg_w = np.median(widths) if widths else 1.0
        max_gap = avg_w * gap_factor
        group = [line[0]]
        for b in line[1:]:
            prev = group[-1]
            gap = b[0] - prev[2]
            if gap <= max_gap:
                group.append(b)
            else:
                x0 = min(bb[0] for bb in group); y0 = min(bb[1] for bb in group)
                x1 = max(bb[2] for bb in group); y1 = max(bb[3] for bb in group)
                merged_words.append((x0,y0,x1,y1))
                group = [b]
        x0 = min(bb[0] for bb in group); y0 = min(bb[1] for bb in group)
        x1 = max(bb[2] for bb in group); y1 = max(bb[3] for bb in group)
        merged_words.append((x0,y0,x1,y1))
    return merged_words

# --------- 6) OCR ----------
def ocr_box(img_bgr, box, psm=7, lang="por"):
    x0,y0,x1,y1 = box
    crop = img_bgr[y0:y1, x0:x1]
    # upscaling ajuda Tesseract em pequenos boxes
    if min(crop.shape[:2]) < 25:
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    try:
        return pytesseract.image_to_string(crop, lang=lang, config=config).strip(), crop
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(crop, lang="eng", config=config).strip(), crop

# --------- 7) Pipeline ----------
def ocr_with_mser(img_bgr):
    gray = preprocess_gray(img_bgr)

    # bin auxiliar: melhora MSER em fundos ruins
    bin_aux = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 9
    )

    regs_pos = mser_regions(gray, delta=5, min_area=60)
    regs_neg = mser_regions(255 - gray, delta=5, min_area=60)
    regs_bin = mser_regions(255 - bin_aux, delta=5, min_area=60)  # opcional

    scored = []
    scored += scored_boxes_from_regions(regs_pos, gray)
    scored += scored_boxes_from_regions(regs_neg, gray)
    scored += scored_boxes_from_regions(regs_bin, gray)

    boxes = suppress_overlaps_scored(scored, iou_thresh=0.30)

    # agrupar em PALAVRAS; psm=7 funciona melhor em palavras/linhas curtas
    word_boxes = group_words(boxes, gap_factor=0.6, y_tol=12)

    results = []
    for bx in word_boxes:
        text, _ = ocr_box(img_bgr, bx, psm=7, lang="por")
        if text:
            results.append({"box": bx, "text": text})

    # se pouco texto, tenta psm=6 (blocos)
    if sum(len(r["text"]) for r in results) < 10:
        results = []
        for bx in word_boxes:
            text, _ = ocr_box(img_bgr, bx, psm=6, lang="por")
            if text:
                results.append({"box": bx, "text": text})

    results = sorted(results, key=lambda r: (r["box"][1], r["box"][0]))
    texto = "\n".join(r["text"] for r in results if r["text"])
    return texto, results
