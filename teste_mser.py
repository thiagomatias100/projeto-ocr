import os, time, cv2, numpy as np
from pdf2image import convert_from_path
from ocr_mser import ocr_with_mser
from PIL import Image
import fitz  # PyMuPDF

# Se quiser desativar totalmente o limite do Pillow (não recomendo sem limites)
# Image.MAX_IMAGE_PIXELS = None

MAX_MP = 20  # teto por página (20 megapixels)
MIN_DPI = 150
MAX_DPI = 400

def pil_to_bgr(pil_img):
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def render_pdf_safe_with_fitz(pdf_path, max_mp=MAX_MP, min_dpi=MIN_DPI, max_dpi=MAX_DPI):
    """Renderiza cada página com um DPI calculado para não passar de max_mp."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        w_pt, h_pt = page.rect.width, page.rect.height  # pontos (72 dpi)
        area_pts = w_pt * h_pt
        # pixels = (area_pts / 72^2) * dpi^2  <=  max_mp*1e6
        # => dpi <= sqrt( max_mp*1e6 * 72^2 / area_pts )
        import math
        dpi_limit = int(math.sqrt((max_mp * 1_000_000) * (72 * 72) / max(1.0, area_pts)))
        dpi = max(min(dpi_limit, max_dpi), min_dpi)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append(bgr)
    doc.close()
    return pages

def ocr_page_bgr(img_bgr):
    texto, resultados = ocr_with_mser(img_bgr)
    # desenhar caixas (opcional)
    for r in resultados:
        x0,y0,x1,y1 = r["box"]
        cv2.rectangle(img_bgr, (x0,y0), (x1,y1), (0,255,0), 1)
    return texto

def main():
    caminho = input("ENTRE COM O ARQUIVO (.PDF/.PNG/.JPG): ").strip('" ')
    if not os.path.exists(caminho):
        print(f"Arquivo '{caminho}' não foi encontrado.")
        return

    inicio_wall = time.time()
    inicio_cpu = time.process_time()

    ext = os.path.splitext(caminho)[1].lower()
    texto_total = []

    if ext == ".pdf":
        print("|°_°| Tentando pdf2image @ 400 DPI...")
        try:
            # DICA: você pode baixar o DPI aqui p/ reduzir o risco:
            paginas_pil = convert_from_path(caminho, dpi=300)
            for i, pil in enumerate(paginas_pil, start=1):
                bgr = pil_to_bgr(pil)
                print(f"|°_°| OCR página {i} (pdf2image)...")
                texto = ocr_page_bgr(bgr)
                texto_total.append(f"\n\n## Página {i}\n\n{texto.strip()}")
        except Exception as e:
            print(f"|°~°| pdf2image falhou: {e}")
            print("|°_°| Fallback: PyMuPDF com limite de pixels por página...")
            pages_bgr = render_pdf_safe_with_fitz(caminho, max_mp=MAX_MP)
            for i, bgr in enumerate(pages_bgr, start=1):
                print(f"|°_°| OCR página {i} (fitz)...")
                texto = ocr_page_bgr(bgr)
                texto_total.append(f"\n\n## Página {i}\n\n{texto.strip()}")

    else:
        img = cv2.imread(caminho)
        if img is None:
            print("|°~°| Não consegui abrir a imagem. Verifique o arquivo.")
            return
        print("|°_°| OCR da imagem...")
        texto = ocr_page_bgr(img)
        texto_total.append(texto.strip())
        # Se quiser ver as caixas na tela:
        # cv2.imshow("Regiões MSER", img); cv2.waitKey(0)

    saida = "\n".join(t for t in texto_total if t)
    print("\n|°_°| O TEXTO FOI EXTRAÍDO:")
    print(saida)

    out_md = os.path.splitext(caminho)[0] + "_ocr.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(saida)

    fim_wall = time.time()
    fim_cpu = time.process_time()
    print(f"\n[OK] Salvo em: {out_md}")
    print(f"|°_°| Tempo de relógio: {fim_wall - inicio_wall:.2f} s")
    print(f"|°_°| Tempo de CPU: {fim_cpu - inicio_cpu:.4f} s")

if __name__ == "__main__":
    main()