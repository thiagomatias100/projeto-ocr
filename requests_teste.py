from pdf2image import convert_from_path
import pytesseract

pdf_path = "testeocr.pdf"

# Converte cada página do PDF em imagem (300 dpi)
pages = convert_from_path(pdf_path, dpi=300)

texto = ""
for i, img in enumerate(pages, 1):
    txt = pytesseract.image_to_string(img, lang="por+eng")
    texto += f"\n\n# Página {i}\n{txt.strip()}"

with open("ocr_local.txt", "w", encoding="utf-8") as f:
    f.write(texto)

print("✅ OCR local concluído. Texto salvo em ocr_local.txt")
