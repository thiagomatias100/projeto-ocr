import base64
import requests
import io
import tempfile
import pyttsx3

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# ---------- Pré-processamento leve p/ OCR ----------
def preprocess_pdf_to_bytes(input_pdf_path, target_dpi=300):
    """
    Lê o PDF, renderiza cada página em alta resolução (target_dpi),
    aplica ajustes leves (grayscale, autocontraste e sharpen) e
    gera um novo PDF otimizado para OCR. Retorna bytes do PDF.
    """
    zoom = target_dpi / 72.0  # 72 DPI é o baseline do PDF
    mat = fitz.Matrix(zoom, zoom)

    src = fitz.open(input_pdf_path)
    out = fitz.open()  # PDF de saída

    for page in src:
        # Renderiza a página como imagem
        pix = page.get_pixmap(matrix=mat, alpha=False)  # sem transparência
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # ---- ajustes leves e seguros p/ OCR ----
        # 1) escala de cinza: remove “ruído” de cor
        img = img.convert("L")

        # 2) autocontraste com recorte suave p/ evitar estourar branco/preto
        img = ImageOps.autocontrast(img, cutoff=2)

        # 3) um toque de nitidez
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=8))

        # 4) leve reforço de contraste/claridade (opcional, mas ajuda muito)
        img = ImageEnhance.Brightness(img).enhance(1.03)
        img = ImageEnhance.Contrast(img).enhance(1.4)

        # Converte de volta para RGB (melhor compatibilidade ao inserir no PDF)
        img_rgb = img.convert("RGB")

        # Cria uma página no PDF de saída do tamanho da imagem
        rect = fitz.Rect(0, 0, img_rgb.width, img_rgb.height)
        outpage = out.new_page(width=rect.width, height=rect.height)

        # Insere a imagem como conteúdo da página
        img_bytes = io.BytesIO()
        img_rgb.save(img_bytes, format="JPEG", quality=90)  # qualidade equilibrada
        img_bytes = img_bytes.getvalue()
        outpage.insert_image(rect, stream=img_bytes)

    # Exporta PDF final para bytes
    pdf_bytes = out.tobytes()
    src.close()
    out.close()
    return pdf_bytes

# ---------- Seu fluxo ORIGINAL, trocando só a origem dos bytes ----------
arquivo_pdf = input("Abrir aquivo: ")

# Tenta pré-processar (se falhar, cai no arquivo original)
try:
    pdf_bytes = preprocess_pdf_to_bytes(arquivo_pdf, target_dpi=300)
except Exception as e:
    print("Aviso: não foi possível pré-processar (seguindo com o original). Detalhe:", e)
    with open(arquivo_pdf, "rb") as f:
        pdf_bytes = f.read()

# Converte para base64
pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

# Monta o payload (mantendo sua estrutura)
payload = {
    "options": {
        "from_formats": ["pdf"],
        "to_formats": ["md"],
        "do_ocr": True,
        "pdf_backend": "dlparse_v4",
        "image_export_mode": "placeholder",
        # Se o servidor aceitar, estas ajudam (caso contrário, serão ignoradas):
        # "ocr_languages": ["por", "eng"],
        # "ocr_dpi": 300
    },
    "file_sources": [
        {
            "base64_string": pdf_base64,
            "filename": "filename.pdf"
        }
    ]
}

# Endpoint do docling-serve
# api_url = "http://localhost:5001/v1alpha/convert/source"
api_url = "http://200.137.132.64:5001/v1alpha/convert/source"

try:
    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "document" in data and "md_content" in data["document"]:
        markdown = data["document"]["md_content"]
        print("Markdown extraído com sucesso:\n")
        print(markdown)
    else:
        print("A conversão foi executada, mas o conteúdo Markdown não foi encontrado.")
        print("Resposta da API:", data)

except requests.exceptions.RequestException as e:
    print("Erro na requisição ao docling-serve:", e)
except ValueError:
    print("Erro ao interpretar a resposta como JSON:")
    print(response.text)

#voz
if "document" in data and "md_content" in data["document"]:
    markdown = data["document"]["md_content"]
    print("Markdown extraído com sucesso:\n")
    print(markdown)

    # --- Leitura em voz ---
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)  # velocidade
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)  # tenta voz feminina se existir
    engine.say(markdown)
    engine.runAndWait()
