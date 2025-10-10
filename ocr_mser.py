import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

def ocr_box(img_bgr, box, psm=7, lang="por"):
    import pytesseract
    x0,y0,x1,y1 = box
    crop = img_bgr[y0:y1, x0:x1]
    config = f"--oem 1 --psm {psm}"
    try:
        return pytesseract.image_to_string(crop, lang=lang, config=config).strip(), crop
    except pytesseract.TesseractError:
        # Se 'por' não estiver instalado, cai para inglês
        return pytesseract.image_to_string(crop, lang="eng", config=config).strip(), crop


def ocr_with_mser(img_bgr):
    gray = preprocess_gray(img_bgr)
    regs1 = mser_regions(gray, delta=5, min_area=60)
    regs2 = mser_regions(255 - gray, delta=5, min_area=60)
    boxes = boxes_from_regions(regs1 + regs2, gray.shape)
    boxes = suppress_overlaps(boxes, iou_thresh=0.3)
    line_boxes = group_lines(boxes, y_tol=12)
    results = []
    for bx in line_boxes:
        text, crop = ocr_box(img_bgr, bx, psm=7, lang="por")
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
