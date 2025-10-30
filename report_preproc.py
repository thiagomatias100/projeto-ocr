#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, cv2, glob, base64, argparse, numpy as np

# --- métricas ---
def vlm(gray):  # Variance of Laplacian (nitidez)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
def entropy(gray):
    h = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel(); p = h/(h.sum()+1e-9); p=p[p>0]
    return float(-(p*np.log2(p)).sum())
def edge_density(gray):
    e = cv2.Canny(gray,80,200); return float((e>0).mean())
def bin_otsu(gray):
    _,b = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); return b

# --- utilidades ---
SUP = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}
def imread(p): return cv2.imread(p, cv2.IMREAD_COLOR)
def to_gray(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def min_side(bgr, m=1400):
    h,w = bgr.shape[:2]; s = (float(m)/min(h,w)) if min(h,w)<m else 1.0
    return cv2.resize(bgr,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC) if s!=1.0 else bgr
def to_b64_png(img_bgr_or_gray):
    if len(img_bgr_or_gray.shape)==2: img=img_bgr_or_gray
    else: img=cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    ok,buf=cv2.imencode(".png", img); assert ok, "Falha encode PNG"
    return "data:image/png;base64,"+base64.b64encode(buf).decode("ascii")

# --- pipelines de pré-processamento ---
def pre_basic(bgr):
    g = to_gray(bgr); g = cv2.createCLAHE(2.0,(8,8)).apply(g)
    g = cv2.addWeighted(g,1.4, cv2.GaussianBlur(g,(0,0),1.0), -0.4,0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def pre_medium(bgr, alpha=1.25, beta=12):
    b = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    g = to_gray(b); return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def pre_doc_clean(bgr):
    bgr = min_side(bgr, 1400)
    g = to_gray(bgr); bg=cv2.GaussianBlur(g,(0,0),21); n=cv2.divide(g,bg,scale=255).astype(np.uint8)
    c = cv2.createCLAHE(2.0,(8,8)).apply(n)
    s = cv2.bilateralFilter(c, d=7, sigmaColor=50, sigmaSpace=50)
    sharp = cv2.addWeighted(s, 2.2, cv2.GaussianBlur(s,(0,0),1.0), -1.2, 0)  # unsharp
    blk, C = 41, 10;  blk += (blk%2==0)
    binimg = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blk,C)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),1)
    return cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)

MODES = {
    "basic": pre_basic,
    "medium": pre_medium,
    "doc-clean": pre_doc_clean,
}

# --- HTML helpers ---
def html_head():
    return """<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Relatório de Pré-processamento (OCR)</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:24px}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:12px 0 28px}
.card{border:1px solid #ddd;padding:10px;border-radius:8px;background:#fff}
.img{width:100%;border:1px solid #eee;border-radius:6px}
h1{margin:0 0 6px} .meta{color:#555;font-size:14px}
.kpi{font-family:monospace;font-size:13px;margin-top:8px}
hr{margin:24px 0}
</style><h1>Relatório de Pré-processamento</h1>
<div class="meta">Comparação: Original × Processada × Binária (Otsu) — com VLM (nitidez), entropia e densidade de bordas.</div><hr>"""
def html_item(name, b64_orig, b64_proc, b64_bin, kpi):
    return f"""<div class="card">
<strong>{name}</strong>
<div class="grid">
  <div><div>Original</div><img class="img" src="{b64_orig}"></div>
  <div><div>Processada</div><img class="img" src="{b64_proc}"></div>
  <div><div>Binária (Otsu)</div><img class="img" src="{b64_bin}"></div>
</div>
<div class="kpi">{kpi}</div>
</div>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Pasta com imagens")
    ap.add_argument("--mode", choices=list(MODES.keys()), default="doc-clean", help="Pipeline: basic | medium | doc-clean")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir,"*.*")))
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in SUP]
    if not paths: print("Nenhuma imagem encontrada."); return

    html = [html_head()]
    proc = MODES[args.mode]

    for p in paths:
        bgr = imread(p); 
        if bgr is None: continue
        bgr_proc = proc(bgr)
        g_orig, g_proc = to_gray(bgr), to_gray(bgr_proc)
        binimg = bin_otsu(g_proc)

        kpi = (f"VLM(orig)={vlm(g_orig):.1f} | VLM(proc)={vlm(g_proc):.1f} | "
               f"Ent(proc)={entropy(g_proc):.2f} | Edges(proc)={edge_density(g_proc):.3f}")

        item = html_item(os.path.basename(p),
                         to_b64_png(bgr), to_b64_png(bgr_proc), to_b64_png(binimg), kpi)
        html.append(item)

    html.append("</html>")
    out = "_report_preproc.html"
    with open(out,"w",encoding="utf-8") as f: f.write("\n".join(html))
    print(f"✅ Relatório gerado: {out}")

if __name__ == "__main__":
    main()
