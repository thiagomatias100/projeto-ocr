"""
UNIVERSIDADE FEDERAL DO MARANHÃO - UFMA
Autor: Thiago Matias da Silva

Projeto de OCR para Identificação de Caracteres e Extração de Textos para Leitura de Tela

Descrição:
Este projeto tem como objetivo aplicar técnicas de Reconhecimento Óptico de Caracteres (OCR) para converter documentos em PDF — sejam nativos ou digitalizados — em texto acessível. 
A finalidade principal é promover acessibilidade para Pessoas com Deficiência (PcD), permitindo que leitores de tela interpretem e transmitam o conteúdo de forma clara e inclusiva.

Funcionalidades previstas:
- Detecção e extração de texto em PDFs nativos sem necessidade de OCR.
- Aplicação de OCR em documentos escaneados para identificar caracteres.
- Pré-processamento de imagens para melhorar a legibilidade.
- Exportação do conteúdo em formatos acessíveis (Markdown, texto simples, áudio).

Finalidade:
Garantir que documentos digitais e digitalizados possam ser lidos de maneira eficiente por tecnologias assistivas, ampliando a inclusão e acessibilidade digital.

"""
import os
import base64
import json
import requests
from PyPDF2 import PdfReader
from docling.document_converter import DocumentConverter
import time


#iniciar contagem de tempo de execução 
inicio = time.time()
for i in range(10**7):
    pass
#iniciar contagem de CPU de execução 
inicio = time.process_time()
for i in range(10**7):
    pass
#Constante com o link da API (UFMA)
#Será usado o docling.Document usando Document_converter
#versão v1Alpha
#API_URL = "http://200.137.132.64:5001/v1alpha/convert/source"
#versão v1
API_URL = "http://200.137.132.64:5005/v1/convert/source"

# Validação de PDF, será feito de forma simples 
# SE conseguir ler o conteudo do PDF ENTÃO é nativo
# consegue extrair => é nativo  
def is_native_pdf(pdf_path: str, min_chars: int = 50) -> bool:
    """
    True se parece nativo (texto extraível com PyPDF2).
    False se escaneado (ou falha na leitura).
    
    """
    try:
        reader = PdfReader(pdf_path)
        extracted = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            extracted.append(t)
        text = "".join(extracted).strip()
        return len(text) >= min_chars
    except Exception as e:
        print(f"[|*_*|detector] Falha PyPDF2: {e}")
        return False

# Extrator para PDF de forma nativa usando API
def extrator_api_nativo(pdf_path: str) -> str:
    """
    Usa a API (docling-serve) SEM OCR (ideal p/ PDF nativo).
    Retorna o markdown (string) ou "" se não obtiver.
    
    """
    # Verivicação da existência do arquivo e validação de extessão .pdf
    try:
        #SE o arquivo nao existir no diretório, Então retornara falso
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo '{pdf_path}' não foi encontrado.")
        #SE a extessão de arquivo nao existir então retornar falso
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Arquivo '{pdf_path}' não é .PDF.")
        #Abrir arquivo no formato de leitura binária e converter para base64   
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        #validação exigida pela API
        payload = {
            "options": {
                "from_formats": ["pdf"],
                "to_formats": ["md"],
                "ocr_lang":["pt","en","es"],
                # OCR desativado p/ nativo para trabalhar como pdf nativo sem imagens 
                "do_ocr": True,
                "pdf_backend": "dlparse_v4",
                "image_export_mode": "placeholder"
            },
            "sources": [
                {"base64_string": pdf_base64, 
                 "filename": os.path.basename(pdf_path),
                 "kind": "file"}#nova versão
            ],

            "target": {"kind": "inbody"} #nova versão
        }

        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print("|°_°| API NATIVO Resposta não-JSON:")
            print(resp.text[:1000])
            return ""

        md = (data.get("document") or {}).get("md_content") or ""
        if md:
            print("|°_°| API NATIVO Markdown extraído com sucesso.\n")
            print(md)
            with open("saida_api.md", "w", encoding="utf-8") as f:
                f.write(md)
        else:
            print("[|°_°| API NATIVO] Sem md_content na resposta (trecho abaixo).")
            print(str(data)[:1000])
        return md
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except requests.exceptions.RequestException as e:
        print("|°~°| API NATIVO Erro de requisição:", e)
        return ""
    except Exception as e:
        print("|°~°| API NATIVO Erro inesperado:", e)
        return ""

def extrator_local_ocr(pdf_path: str) -> str:
    """
    Usa Docling local (com OCR quando necessário).
    Retorna o markdown (string) ou "" se não obtiver.
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo '{pdf_path}' não foi encontrado.")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Arquivo '{pdf_path}' não é .PDF.")

        converter = DocumentConverter()  # seu ambiente já registra OCR engines
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown() or ""
        if md:
            print("|°_°|LOCAL OCR Markdown extraído com sucesso.\n")
            print(md)
            with open(f"{pdf_path[:-4]}saida_local_ocr.md", "w", encoding="utf-8") as f:
                f.write(md)
        else:
            print("|°~°|LOCAL OCR Conversão retornou vazio.")
        return md
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except Exception as e:
        print("|°~°|LOCAL OCR Erro ao processar com Docling:", e)
        return ""

# "PIPELINE":Controlador ou "orquestrador" 
def processar_pdf(pdf_path: str, min_chars: int = 50):
    """
    Estratégia solicitada:
      1) Tenta OCR via API primeiro (sempre).
      2) Se vier vazio/der erro, faz fallback para OCR local (Docling).
    """
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("|°~°| Forneça um caminho válido para um .pdf existente.")
        return ""

    print("|°_°| Tentando API (OCR no servidor)...")
    md = extrator_api_nativo(pdf_path)
    if md:
        return md

    print("|°_°| Fallback: tentando LOCAL (OCR Docling)...")
    md = extrator_local_ocr(pdf_path)
    if md:
        return md

    print("|°~°| Falhou API e LOCAL. Verifique conectividade com a API e engines OCR locais.")
    return ""


if __name__ == "__main__":
    pdf_path = input("ENTRE COM O NOME DO ARQUIVO: ").strip().strip('"').strip("'")
    processar_pdf(pdf_path, min_chars=50)

#Inalizar contagem da execução
fim = time.time()
print(f"|°_°| Tempo de execução: {fim - inicio:.2f} segundos")
#Medir tempo de CPU
fim = time.process_time()
print(f"|°_°| Tempo de CPU: {fim - inicio:.4f} segundos")