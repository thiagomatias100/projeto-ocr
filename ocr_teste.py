# api_ocr_novo.py
#refazer
import os, json, base64, requests
from pathlib import Path

API_URL = "http://200.137.132.64:5005/v1alpha/convert/source"

def read_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_payload_novo(pdf_path: Path) -> dict:
    """
    Payload NOVO compatível com sua especificação:
      - options: campos ampliados (ocr_engine, ocr_lang, pdf_backend etc.)
      - sources: lista com base64 + filename + kind="file"
      - target: inbody (resposta volta no corpo)
    Ajuste os campos conforme sua necessidade.
    """
    return {
        "options": {
            "from_formats": [
                "docx","pptx","html","image","pdf","asciidoc","md","csv","xlsx",
                "xml_uspto","xml_jats","mets_gbs","json_docling","audio","vtt"
            ],
            "to_formats": ["md"],

            # === OCR / Parsing ===
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,               # deixe False se quiser respeitar pdf_backend
            "ocr_engine": "easyocr",          # "tesseract_cli" | "easyocr" | "rapidocr" (varia por build)
            "ocr_lang": ["fr","de","es","en"],# no new-serve costuma ser LISTA
            "pdf_backend": "pypdfium2",       # "ocr" para OCR total; "pypdfium2"/"dlparse_v4" para parser

            # === Tabelas ===
            "table_mode": "fast",             # "fast" | "accurate" (dependendo do build)
            "table_cell_matching": True,
            "do_table_structure": True,

            # === Pipeline, páginas e timeouts ===
            "pipeline": "standard",
            "page_range": [1, 9223372036854776000],
            "document_timeout": 604800,       # 7 dias
            "abort_on_error": False,

            # === Imagens e MD ===
            "include_images": True,
            "images_scale": 2,
            "md_page_break_placeholder": "<!-- page-break -->",

            # === Enriquecimentos (desligados por padrão) ===
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,

            # === (Opcional) Configs de VLM/descrição de imagens ===
            "picture_description_area_threshold": 0.05,
            "picture_description_local": {
                "generation_config": {"do_sample": False, "max_new_tokens": 200},
                "prompt": "Describe this image in a few sentences.",
                "repo_id": "ibm-granite/granite-vision-3.2-2b",
            },
            "picture_description_api": {
                "concurrency": 1,
                "headers": {},
                "params": {"model": "granite3.2-vision:2b"},
                "prompt": "Describe this image in a few sentences.",
                "timeout": 20,
                "url": "http://localhost:1234/v1/chat/completions",
            },
            "vlm_pipeline_model": "granite_docling",
            "vlm_pipeline_model_local": {
                "extra_generation_config": {"skip_special_tokens": False},
                "inference_framework": "transformers",
                "prompt": "Convert this page to docling.",
                "repo_id": "ibm-granite/granite-docling-258M",
                "response_format": "doctags",
                "scale": 2,
                "transformers_model_type": "automodel-imagetexttotext",
            },
            "vlm_pipeline_model_api": {
                "concurrency": 1,
                "headers": {},
                "params": {"model": "ibm-granite/granite-docling-258M-mlx"},
                "prompt": "Convert this page to docling.",
                "response_format": "doctags",
                "scale": 2,
                "timeout": 60,
                "url": "http://localhost:1234/v1/chat/completions",
            },
        },
        "sources": [
            {
                "base64_string": read_b64(pdf_path),
                "filename": pdf_path.name,
                "kind": "file",
            }
        ],
        "target": {"kind": "inbody"},
    }

def build_payload_legacy(pdf_path: Path) -> dict:
    """
    Fallback LEGADO (caso o novo formato não seja aceito).
    Usa 'file_sources' e não tem 'target'.
    """
    return {
        "options": {
            "from_formats": ["pdf"],
            "to_formats": ["md"],
            "do_ocr": True,
            "pdf_backend": "ocr",  # OCR total, compatível com builds antigos
            "image_export_mode": "placeholder",
        },
        "file_sources": [
            {
                "base64_string": read_b64(pdf_path),
                "filename": pdf_path.name,
            }
        ],
    }

def call_api(payload: dict, timeout: int = 240) -> dict:
    resp = requests.post(API_URL, json=payload, timeout=timeout)
    # Se não for 2xx, levanta para tratarmos erro/fallback
    resp.raise_for_status()
    try:
        return resp.json()
    except requests.JSONDecodeError:
        raise RuntimeError(f"Resposta não-JSON da API: {resp.text[:1000]}")

def extract_markdown(api_json: dict) -> str:
    # novo/antigo costumam devolver 'document.md_content'
    doc = api_json.get("document") or {}
    return doc.get("md_content") or ""

def convert_with_fallback(pdf_path: str) -> str:
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf}")

    # 1) tenta payload NOVO
    try:
        print(f"|°_°| Enviando payload NOVO para {API_URL} …")
        data = call_api(build_payload_novo(pdf))
        md = extract_markdown(data)
        if md:
            print("|°_°| OK (NOVO).")
            with open("saida_api.md", "w", encoding="utf-8") as f:
                f.write(md)
            return md
        else:
            print("|°~°| NOVO sem md_content — tentando LEGADO…")
    except requests.HTTPError as e:
        # 400/422/415 etc. → tenta LEGADO
        print(f"|°~°| Erro HTTP no NOVO: {e} — tentando LEGADO…")
    except Exception as e:
        print(f"|°~°| Falha no NOVO: {e} — tentando LEGADO…")

    # 2) fallback LEGADO
    data = call_api(build_payload_legacy(pdf))
    md = extract_markdown(data)
    if not md:
        raise RuntimeError("API respondeu mas não retornou md_content em nenhum formato (NOVO/LEGADO).")
    print("|°_°| OK (LEGADO).")
    with open("saida_api.md", "w", encoding="utf-8") as f:
        f.write(md)
    return md

if __name__ == "__main__":
    import sys
    try:
        pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("PDF: ").strip().strip('"').strip("'")
        md = convert_with_fallback(pdf_path)
        print("\n--- INÍCIO DO MARKDOWN ---\n")
        print(md[:2000])  # mostra só um trecho
        print("\n--- FIM (arquivo completo em: saida_api.md) ---")
    except KeyboardInterrupt:
        print("\nCancelado pelo usuário.")
    except Exception as e:
        print("[ERRO]", e)
