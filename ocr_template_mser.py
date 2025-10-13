import cv2
import numpy as np
import pytesseract
import json, glob, re
from pathlib import Path

# --------- AJUSTE WINDOWS (se necessário) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===================================================
# SUA PIPELINE (pré-processamento + MSER) - intacta
# ===================================================

def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return sharp

def mser_regions(gray, delta=5, min_area=60, max_area_ratio=0.25):
    h, w = gray.shape
    max_area = int(h*w*max_area_ratio)
    mser = cv2.MSER_create()
    mser.setDelta(delta)
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, _ = mser.detectRegions(gray)
    return regions

def boxes_from_regions(regions, img_shape):
    h, w = img_shape
    boxes = []
    for pts in regions:
        x,y,ww,hh = cv2.boundingRect(pts)
        area = ww*hh
        if area < 100 or area > (w*h*0.25):
            continue
        aspect = ww / max(hh,1)
        if aspect < 0.2 or aspect > 8:
            continue
        pad = 2
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
        boxes.append((x0,y0,x1,y1))
    return boxes

def suppress_overlaps(boxes, iou_thresh=0.3):
    rects = np.array(boxes)
    if len(rects) == 0: return []
    x1 = rects[:,0]; y1 = rects[:,1]; x2 = rects[:,2]; y2 = rects[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = np.argsort(areas)
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
    return [boxes[i] for i in keep]

def group_lines(boxes, y_tol=10):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    current = [boxes[0]]
    for b in boxes[1:]:
        prev = current[-1]
        same_line = abs(b[1] - prev[1]) <= y_tol
        if same_line:
            current.append(b)
        else:
            lines.append(sorted(current, key=lambda r: r[0]))
            current = [b]
    lines.append(sorted(current, key=lambda r: r[0]))
    merged = []
    for line in lines:
        x0 = min(b[0] for b in line); y0 = min(b[1] for b in line)
        x1 = max(b[2] for b in line); y1 = max(b[3] for b in line)
        merged.append((x0,y0,x1,y1))
    return merged

def ocr_box(img_bgr, box, psm=7, lang="por", whitelist=None):
    x0,y0,x1,y1 = box
    crop = img_bgr[y0:y1, x0:x1]
    # Usa seu preprocess + um threshold leve
    gray = preprocess_gray(crop)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)

    config = f"--oem 1 --psm {psm} -l {lang}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    try:
        return pytesseract.image_to_string(th, config=config).strip(), th
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(th, lang="eng", config=config).strip(), th

def ocr_with_mser(img_bgr):
    gray = preprocess_gray(img_bgr)
    regs1 = mser_regions(gray, delta=5, min_area=60)
    regs2 = mser_regions(255 - gray, delta=5, min_area=60)
    boxes = boxes_from_regions(regs1 + regs2, gray.shape)
    boxes = suppress_overlaps(boxes, iou_thresh=0.3)
    line_boxes = group_lines(boxes, y_tol=12)
    results = []
    for bx in line_boxes:
        text, _ = ocr_box(img_bgr, bx, psm=7, lang="por")
        if text:
            results.append({"box": bx, "text": text})
    if sum(len(r["text"]) for r in results) < 10:
        results = []
        for bx in line_boxes:
            text, _ = ocr_box(img_bgr, bx, psm=6, lang="por")
            if text:
                results.append({"box": bx, "text": text})
    results = sorted(results, key=lambda r: (r["box"][1], r["box"][0]))
    texto = "\n".join(r["text"] for r in results if r["text"])
    return texto, results

# ===================================================
# CAMADA DE TEMPLATE (alinhamento + ROIs + validação)
# ===================================================

def valida_cpf(s: str) -> bool:
    digits = re.sub(r"\D", "", s or "")
    if len(digits) != 11 or digits == digits[0]*11:
        return False
    def dv(cpf, p):
        soma = sum(int(cpf[i]) * (p - i) for i in range(p-1))
        r = (soma * 10) % 11
        return 0 if r == 10 else r
    d1 = dv(digits, 10)
    d2 = dv(digits, 11)
    return int(digits[9]) == d1 and int(digits[10]) == d2

def load_template(json_path: Path):
    tpl = json.loads(Path(json_path).read_text(encoding="utf-8"))
    img_path = Path(tpl["template_image"])
    tpl_img = cv2.imread(str(img_path))
    if tpl_img is None:
        raise FileNotFoundError(f"Falha ao carregar template: {img_path}")
    w, h = tpl["size"]["width"], tpl["size"]["height"]
    tpl_img = cv2.resize(tpl_img, (w, h))
    tpl["template_bgr"] = tpl_img
    return tpl

def compute_homography_and_warp(img_bgr, template_bgr, out_size):
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 10:
        return None, 0
    matches = sorted(matches, key=lambda m: m.distance)[:600]
    src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None:
        return None, 0
    W, Ht = out_size
    warped = cv2.warpPerspective(img_bgr, H, (W, Ht))
    return warped, inliers

def choose_best_template(input_bgr, templates):
    best_tpl, best_warp, best_inliers = None, None, -1
    for tpl in templates:
        w, h = tpl["size"]["width"], tpl["size"]["height"]
        warped, inl = compute_homography_and_warp(input_bgr, tpl["template_bgr"], (w, h))
        if warped is not None and inl > best_inliers:
            best_tpl, best_warp, best_inliers = tpl, warped, inl
    return best_tpl, best_warp, best_inliers

def ocr_roi_with_fallback(warped_bgr, field):
    x, y, w, h = field["roi"]
    roi = warped_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return "", False, "ROI vazia"

    # 1) Tenta OCR direto com sua pipeline no recorte
    text, _ = ocr_box(
        warped_bgr,
        (x, y, x+w, y+h),
        psm=field.get("psm", 7),
        lang=field.get("lang", "por"),
        whitelist=field.get("whitelist")
    )

    # 2) Regex (se houver)
    rgx = field.get("regex")
    if rgx and text:
        m = re.search(rgx, text)
        if m:
            text = m.group(0)

    # 3) Validação (ex.: CPF)
    ok = True
    if field.get("validate") == "cpf" and text:
        ok = valida_cpf(text)

    # 4) Fallback com MSER (NESTA ROI) se não passou regex/validação
    if not text or (rgx and not re.search(rgx, text)) or not ok:
        # Rode o seu OCR por MSER apenas dentro da ROI:
        mser_text, _ = ocr_with_mser(roi)
        if mser_text:
            text = mser_text.strip()
            # aplica regex de novo
            if rgx:
                m = re.search(rgx, text)
                if m:
                    text = m.group(0)
            if field.get("validate") == "cpf":
                ok = valida_cpf(text)
            else:
                ok = True if (not rgx or re.search(rgx, text)) else False

    return text, ok, None if ok else "regex/validação falhou"

def extract_fields_from_template(warped_bgr, tpl):
    out = {}
    for f in tpl["fields"]:
        txt, ok, reason = ocr_roi_with_fallback(warped_bgr, f)
        out[f["key"]] = {"text": txt, "ok": ok, "reason": reason}
    return out

def run_ocr_template_mser(image_path: str, templates_dir: str = "templates"):
    inp = cv2.imread(image_path)
    if inp is None:
        raise ValueError(f"Não consegui abrir {image_path}")

    # Carrega templates
    tpl_jsons = glob.glob(str(Path(templates_dir) / "*.json"))
    templates = [load_template(Path(p)) for p in tpl_jsons]
    if not templates:
        raise RuntimeError("Nenhum template encontrado em 'templates/'.")

    # Escolhe melhor template
    best_tpl, warped, inliers = choose_best_template(inp, templates)
    if best_tpl is None:
        raise RuntimeError("Falha ao alinhar com os templates.")
    results = extract_fields_from_template(warped, best_tpl)

    return {
        "template": best_tpl["name"],
        "inliers": inliers,
        "results": results
    }

if __name__ == "__main__":
    import sys, json as pyjson
    if len(sys.argv) < 2:
        print("Uso: python ocr_template_mser.py <imagem_entrada>")
        sys.exit(1)
    out = run_ocr_template_mser(sys.argv[1], templates_dir="templates")
    print(pyjson.dumps(out, ensure_ascii=False, indent=2))
