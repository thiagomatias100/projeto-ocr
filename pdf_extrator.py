
import base64
import requests
import os
from PyPDF2 import PdfReader
from docling.document_converter import DocumentConverter


pdf_path = input("ENTRE COM O NOME DO ARQUIVO: ")
def extrator(pdf_path):

    API_URL = "http://200.137.132.64:5001/v1alpha/convert/source"


    try:   
        #importar aquivo.pdf
       # arquivo_pdf = input("Abrir aquivo: ")
        arquivo_pdf = pdf_path
        if not os.path.exists(arquivo_pdf) or not arquivo_pdf.lower().endswith('.pdf'):
            raise FileNotFoundError (f"Erro arquivo '{arquivo_pdf}' não foi encontrado")  

        if not arquivo_pdf.lower().endswith('.pdf'):
            raise ValueError (f"Erro: Arquivo '{arquivo_pdf}' não é do tipo .PDF")   

        # Abrir o arquivo.pdf local e converter para base64       
            
        with open(arquivo_pdf,"rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        #passar os parametros para API com o OCR ativado
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

        #chamar a API
        # Endpoint do docling-serve
        #api_url = "http://localhost:5001/v1alpha/convert/source"



        response = requests.post(API_URL, json=payload)
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
    except FileNotFoundError as e:
        print(e)    
    except requests.exceptions.RequestException as e:
        print("Erro na requisição ao docling-serve:", e)
    except ValueError:
        print("Erro ao interpretar a resposta como JSON:")
        print(response.text)


#extrator 2

def extrator2(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Erro: arquivo '{pdf_path}' não foi encontrado")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Erro: arquivo '{pdf_path}' não é do tipo .PDF")

        # inicializa o conversor
        converter = DocumentConverter()

        # converte o PDF local
        result = converter.convert(pdf_path)

        # exporta para Markdown
        markdown = result.document.export_to_markdown()

        print("Markdown extraído com sucesso:\n")
        print(markdown)

        # opcional: salvar em arquivo
        with open("saida.md", "w", encoding="utf-8") as f:
            f.write(markdown)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("Erro ao processar com o Docling:", e)




#verificar as procidencias do pdf
def is_native_pdf(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
                #print(text)
            # Se houver texto extraído, é um PDF nativo
            print("extartor1")
            extrator(pdf_path) 
            return bool(text)
        except Exception as e:
            print("extartor2")
            # Se ocorrer um erro na extração, provavelmente é um PDF escaneado ou corrompido
            print(f"Erro ao ler PDF: {e}")
            extrator2(pdf_path)
            return False
        




if __name__ == "__main__":
    
     is_native_pdf(pdf_path)
