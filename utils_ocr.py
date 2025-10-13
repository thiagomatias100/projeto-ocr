# utils_ocr.py
import cv2, numpy as np, pytesseract, os
from pytesseract import Output

# se você já usa config_ocr.py, essa import resolve caminho do tesseract/tessdata
try:
    import config_ocr  # opcional
except Exception:
    pass

def _deskew_with_hough(img_bgr, min_line_len=120, max_line_gap=12):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None or len(lines)==0: 
        return img_bgr, 0.0
    angs=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        dx, dy = x2-x1, y2-y1
        if dx==0 and dy==0: continue
        a = np.degrees(np.arctan2(dy, dx))
        if a < -90: a += 180
        if a >= 90: a -= 180
        angs.append(a)
    if not angs: return img_bgr, 0.0
    angle = float(np.median(angs))
    h,w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    rot = cv2.warpAffine(img_bgr, M, (w,h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return rot, angle

def _remove_shadow(gray):
    bg = cv2.medianBlur(gray, 31)
    return cv2.divide(gray, bg, scale=255)

def _prep_variants(img_bgr, scale=2.5):
    h,w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _remove_shadow(gray)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    bin_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bin_adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 9)
    inv_otsu = cv2.bitwise_not(bin_otsu)
    inv_adap = cv2.bitwise_not(bin_adap)
    return [sharp, bin_otsu, bin_adap, inv_otsu, inv_adap]

def _tesseract_try(img, lang="por+eng", psm=6, extra=None):
    cfg = f"--oem 1 --psm {psm}"
    if extra: cfg += " " + extra
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=Output.DICT)
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(img, lang="eng", config=cfg, output_type=Output.DICT)
    words = [w for w in data["text"] if isinstance(w, str) and w.strip()]
    text = " ".join(words)
    confs = [float(c) for c in data["conf"] if str(c).replace('.','',1).isdigit()]
    conf = sum(confs)/len(confs) if confs else -1.0
    return text, conf

def ocr_try_hard_bgr(img_bgr, lang_default="por+eng"):
    img_corr, ang = _deskew_with_hough(img_bgr)
    variants = _prep_variants(img_corr, scale=2.5)
    tries = []
    for v in variants:
        for psm in (6, 11, 7):  # bloco | sparse | linha
            txt, conf = _tesseract_try(v, lang=lang_default, psm=psm,
                                       extra="-c preserve_interword_spaces=1")
            tries.append((conf, len(txt), txt))
    tries.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_conf, _, best_txt = tries[0]
    return best_txt, best_conf, ang

def rasterize_pdf_first_page(pdf_path, dpi=400):
    import fitz, numpy as np, cv2  # PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
