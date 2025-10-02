import os
from docling.document_converter import DocumentConverter


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


# usar em caso de problema do extrator principal
extrator2()
