import requests
import urllib3
import base64

# Desativa o aviso de certificado inseguro (https://www.ufma.br)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#pdf_url = "https://portalpadrao.ufma.br/proen/digrad/ingresso-sisu/2025.1/editais-sisu/arquivos-editais-sisu/nota-informativa.pdf"
pdf_url = "http://localhost/testeocrs/testeocr.pdf"
    
with open("testeocr.pdf", "rb") as f:           # sempre "rb" para binário
    pdf_base64  = base64.b64encode(f.read()).decode("utf-8")


try:
    # Baixando o PDF ignorando problemas de certificado SSL
    response = requests.get(pdf_url, verify=False)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print("Erro ao baixar o PDF:", e)
    exit()

# Codifica o PDF em base64
pdf_base64 = base64.b64encode(response.content).decode("utf-8")

# Payload esperado pela API docling-serve
payload = {
    "options": {
        "from_formats": ["pdf"],
        "to_formats": ["md"],
        "do_ocr": True,
        "pdf_backend": "dlparse_v4",
        "image_export_mode": "placeholder"
    },
    "file_sources": [
        {
            "base64_string": pdf_base64,
            "filename": "filename.pdf"
        }
    ]
}

# Endpoint do docling-serve
#api_url = "http://localhost:5001/v1alpha/convert/source"
api_url = "http://200.137.132.64:5001/v1alpha/convert/source"

try:
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    data = response.json()

    # Verifica se o markdown foi retornado corretamente
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
