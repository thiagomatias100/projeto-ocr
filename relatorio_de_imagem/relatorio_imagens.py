"""
Relatório automático de imagens para análise de OCR
Autor: Thiago Matias da Silva (UFMA)
Descrição:
    Gera um relatório com informações técnicas (OSD, tamanho, resolução, confiança)
    para avaliar a qualidade de imagens destinadas ao OCR.
"""

import cv2
import pytesseract
import os
import csv
from datetime import datetime
from PIL import Image

# Caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Pasta com imagens
PASTA_IMAGENS = "input_imgs"
# Arquivo de saída
ARQUIVO_RELATORIO = "relatorio_imagens.csv"

# Cabeçalho do CSV
cabecalho = [
    "Arquivo", "Largura (px)", "Altura (px)", "Resolução estimada (DPI)",
    "Orientação (graus)", "Rotacionar", "Confiança orientação",
    "Script", "Confiança script", "Qtd. caracteres", "Classificação", "Data/Hora"
]

linhas = []

# Cria pasta se não existir
if not os.path.exists(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS)
    print(f"Pasta '{PASTA_IMAGENS}' criada. Coloque as imagens lá e rode novamente.")
    exit()

# Percorre todas as imagens da pasta
for nome in os.listdir(PASTA_IMAGENS):
    if not nome.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        continue

    caminho = os.path.join(PASTA_IMAGENS, nome)
    img = cv2.imread(caminho)
    h, w = img.shape[:2]

    # converte p/ PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Obtém número de caracteres reconhecidos (teste de densidade)
    texto = pytesseract.image_to_string(pil_img)
    qtd_chars = len(texto.strip())

    # Detecta OSD (orientação)
    try:
        osd_data = pytesseract.image_to_osd(pil_img)
        dados = dict([line.split(": ") for line in osd_data.split("\n") if ": " in line])
    except Exception as e:
        dados = {}
        print(f"Erro OSD em {nome}: {e}")

    # Extrai dados do OSD
    orientacao = dados.get("Orientation in degrees", "N/A")
    rotacionar = dados.get("Rotate", "N/A")
    conf_orient = dados.get("Orientation confidence", "0")
    script = dados.get("Script", "N/A")
    conf_script = dados.get("Script confidence", "0")

    # Classificação simples da imagem
    if qtd_chars > 500 and float(conf_orient) > 50:
        qualidade = "Boa"
    elif qtd_chars > 100:
        qualidade = "Média"
    else:
        qualidade = "Ruim"

    # Adiciona linha
    linhas.append([
        nome, w, h, "≈300", orientacao, rotacionar, conf_orient,
        script, conf_script, qtd_chars, qualidade, datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ])

# Salva no CSV
with open(ARQUIVO_RELATORIO, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(cabecalho)
    writer.writerows(linhas)

print(f"✅ Relatório gerado com sucesso: {ARQUIVO_RELATORIO}")
