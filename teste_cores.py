import os
from pathlib import Path
import cv2
import numpy as np

# =========================
# SUAS FUNÇÕES (inalteradas)
# =========================
def preprocess_gray(img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return sharp

def mser_regions(gray, delta=5, min_area=120, max_area_ratio=0.12):
    h, w = gray.shape
    mser = cv2.MSER_create()
    mser.setDelta(delta)                          # ↑ menos redundância
    mser.setMinArea(min_area)                    # ↑ corta ruído miúdo
    mser.setMaxArea(int(h*w*max_area_ratio))     # ↓ evita regiões gigantes
    mser.setMaxVariation(0.15)                   # ↓ estabilidade mais rígida
    mser.setMinDiversity(0.7)                    # ↑ força diversidade
    mser.setEdgeBlurSize(5)                      # suaviza antes de analisar
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
# =========================

# ---------- helpers de debug/visual ----------
def draw_boxes(img_bgr, boxes, color=(0,255,0), thickness=1):
    dbg = img_bgr.copy()
    for (x0,y0,x1,y1) in boxes:
        cv2.rectangle(dbg, (x0,y0), (x1,y1), color, thickness)
    return dbg

def stack_h(images, pad=4):
    # junta imagens (grayscale ou BGR) lado a lado
    # converte cinza->BGR para empilhar
    conv = []
    max_h = max(im.shape[0] for im in images)
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[0] != max_h:
            scale = max_h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1]*scale), max_h), interpolation=cv2.INTER_AREA)
        conv.append(im)
    gap = 255*np.ones((max_h, pad, 3), dtype=np.uint8)
    row = conv[0]
    for im in conv[1:]:
        row = np.hstack([row, gap, im])
    return row

def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# ---------------- main ----------------
if __name__ == "__main__":
    path = input("Imagem (.png/.jpg): ").strip().strip('"').strip("'")
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("Não consegui ler a imagem.")
    img  = ensure_bgr(img)

    out_dir = Path("debug_seq")
    out_dir.mkdir(exist_ok=True)
    base = Path(path).stem

    # 01) original
    cv2.imwrite(str(out_dir / f"{base}_01_original.png"), img)

    # 02–05) etapas do preprocess
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / f"{base}_02_gray.png"), gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    cv2.imwrite(str(out_dir / f"{base}_03_clahe.png"), clahe)

    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    cv2.imwrite(str(out_dir / f"{base}_04_blur.png"), blur)

    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    cv2.imwrite(str(out_dir / f"{base}_05_sharp.png"), sharp)

    # 06–08) MSER normal/invertido + máscaras
    regs1 = mser_regions(sharp, delta=5, min_area=60)          # texto escuro
    regs2 = mser_regions(255 - sharp, delta=5, min_area=60)    # texto claro

    mask1 = np.zeros_like(sharp); 
    for pts in regs1:
        cv2.drawContours(mask1, [pts], -1, 255, 1)
    cv2.imwrite(str(out_dir / f"{base}_06_mser_mask_normal.png"), mask1)

    mask2 = np.zeros_like(sharp);
    for pts in regs2:
        cv2.drawContours(mask2, [pts], -1, 255, 1)
    cv2.imwrite(str(out_dir / f"{base}_07_mser_mask_invertido.png"), mask2)

    # 09–11) caixas de regiões (cruas), NMS e linhas
    boxes_all = boxes_from_regions(regs1 + regs2, sharp.shape)
    dbg_boxes_all = draw_boxes(img, boxes_all, (0,255,255), 1)
    cv2.imwrite(str(out_dir / f"{base}_08_boxes_cruas.png"), dbg_boxes_all)

    boxes_nms = suppress_overlaps(boxes_all, iou_thresh=0.3)
    dbg_boxes_nms = draw_boxes(img, boxes_nms, (0,255,0), 1)
    cv2.imwrite(str(out_dir / f"{base}_09_boxes_nms.png"), dbg_boxes_nms)

    line_boxes = group_lines(boxes_nms, y_tol=12)
    dbg_line_boxes = draw_boxes(img, line_boxes, (255,0,0), 2)
    cv2.imwrite(str(out_dir / f"{base}_10_line_boxes.png"), dbg_line_boxes)

    # 12) COMBOS lado-a-lado: pré-processamento e MSER
    combo_pre  = stack_h([gray, clahe, blur, sharp], pad=6)
    cv2.imwrite(str(out_dir / f"{base}_11_combo_preprocess.png"), combo_pre)

    combo_mser = stack_h([
        ensure_bgr(img),
        cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR),
        dbg_boxes_nms,
        dbg_line_boxes
    ], pad=6)
    cv2.imwrite(str(out_dir / f"{base}_12_combo_mser.png"), combo_mser)

    print("Imagens salvas em:", out_dir.resolve())
    print("Sequência:")
    for i in range(1, 13):
        print(f"  {i:02d} -> {base}_{i:02d}_*.png")
