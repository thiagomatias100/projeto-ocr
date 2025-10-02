from docling.document_converter import DocumentConverter
from pathlib import Path
import os

os.environ["HF_TOKEN"] = "hf_"
source = Path(r"C:\Projetos\docling\rg.pdf")
converter = DocumentConverter()
result = converter.convert(source)

# Export to markdown
markdown_content = result.document.export_to_markdown()
print(markdown_content)