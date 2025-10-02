from PyPDF2 import PdfReader
from pdf_extrator import extrator
from docling.document_converter import DocumentConverter

pdf_path = input("ENTRE COM O NOME DO ARQUIVO EM PDF: ")   
def is_native_pdf(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
              #print(text)
            # Se houver texto extraído, é um PDF nativo
            return bool(text)
        except Exception as e:
            # Se ocorrer um erro na extração, provavelmente é um PDF escaneado ou corrompido
            print(f"Erro ao ler PDF: {e}")
            return False
        
if is_native_pdf(pdf_path):  
   print("O PDF é nativo.")
   extrator(pdf_path)  
else:
  print("O PDF é escaneado ou há um problema na leitura.")
  extrator2(pdf_path)


#extrator2 sera para caso ocorra problemas na leitura do extrator 1
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


if __name__ == "__main__":
    
    if is_native_pdf == False:
        from pdf_extrator2 import extrator2   
        print("usaremos outro modo")
        extrator2(pdf_path)
    else:
       
       print("PDF Nativo ou escaneado de boa qualidade ")
       extrator(pdf_path)