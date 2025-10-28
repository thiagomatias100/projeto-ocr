import os
import re
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List
from paddleocr import PPStructureV3 # O motor de analise de documentos robusto

# --- Configuração do PaddleOCR (APENAS UM MOTOR) ---
try:
    # use_textline_orientation é o parâmetro correto para 3.x
    PPSTRUCTURE_ENGINE = PPStructureV3(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False
    )
    # Se a inicialização falhar, a exceção será capturada
except Exception as e:
    print(f"ERRO CRÍTICO: Falha ao inicializar PPStructureV3. Verifique a instalação do PaddlePaddle.")
    print(f"Detalhes: {e}")
    PPSTRUCTURE_ENGINE = None
    exit() # Interrompe se o motor principal não iniciar

# -----------------------------------------------------------------------------
# |°¿°| FUNÇÕES AUXILIARES NECESSÁRIAS
# -----------------------------------------------------------------------------

def extrair_dados_pessoais(texto_ocr: str) -> Dict[str, Optional[str]]:
    """
    Usa Expressões Regulares para encontrar padrões comuns em documentos brasileiros.
    """
    dados = {
        "cpf": None,
        "rg": None,
        "nome_completo": None,
        "data_nascimento": None
    }
    
    texto_limpo = texto_ocr.replace('\n', ' ').replace('\r', ' ').upper()
    
    # Extração de CPF (Busca por "CPF" ou 11 dígitos)
    match_cpf = re.search(r'C[PF]?[F]?\s*[.:]?\s*(\d{2,3}[.\s-]?\d{3}[.\s-]?\d{3}[.\s-]?\d{2})', texto_limpo)
    if match_cpf:
        cpf_limpo = re.sub(r'[\.\s-]', '', match_cpf.group(1))
        if len(cpf_limpo) == 11 and cpf_limpo.isdigit():
            dados["cpf"] = cpf_limpo[:3] + '.' + cpf_limpo[3:6] + '.' + cpf_limpo[6:9] + '-' + cpf_limpo[9:]
    
    # Extração de RG (Busca por "RG" ou "REGISTRO GERAL" + número)
    match_rg = re.search(r'(R[EGISTRO\s]*G[ERAL]*\s*[.:]?\s*(\d{1,2}[.\s-]?\d{3}[.\s-]?\d{3}[.\s-]?[\dX]))', texto_limpo)
    if match_rg:
        rg_limpo = re.sub(r'[\.\s-]', '', match_rg.group(2))
        dados["rg"] = rg_limpo
        
    # Extração de Data de Nascimento
    match_data = re.search(r'(DATA DE NASCIMENTO|NASCIDO EM|NASC\.\s*[.:]?)\s*(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})', texto_limpo)
    if match_data:
        dados["data_nascimento"] = match_data.group(2)
        
    # Extração de Nome (Simplificado)
    match_nome = re.search(r'(NOME\s*COMPLETO|NOME\s*:\s*)([A-Z\s]{5,100})', texto_limpo)
    if match_nome:
        dados["nome_completo"] = match_nome.group(2).strip()
        
    return dados


# -----------------------------------------------------------------------------
# |°¿°| FUNÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

def processar_documento_ppstructure(caminho_arquivo: str) -> str:
    """
    Processa o documento usando PPStructureV3 e retorna o texto completo em Markdown.
    """
    if PPSTRUCTURE_ENGINE is None:
        return ""
    
    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: Arquivo não encontrado: {caminho_arquivo}")
        return ""

    print(f"\n➡️ Processando documento com PPStructureV3: {os.path.basename(caminho_arquivo)}")
    
    try:
        # PPStructureV3 é o motor mais robusto e aceita o caminho do arquivo diretamente
        output = PPSTRUCTURE_ENGINE.predict(input=caminho_arquivo)
        
        markdown_text = ""
        
        for res in output:
            # Pega o texto formatado como Markdown
            markdown_text += res.to_markdown() + "\n"
            
        return markdown_text.strip()
        
    except Exception as e:
        print(f"❌ Erro durante o processamento do PPStructureV3: {e}")
        return ""

# -----------------------------------------------------------------------------
# |°¿°| EXECUÇÃO
# -----------------------------------------------------------------------------

def main():
    path = input("ENTRE COM O CAMINHO DO ARQUIVO (PDF ou IMAGEM): ").strip().strip('"').strip("'")
    
    texto_bruto = processar_documento_ppstructure(path)
    
    if texto_bruto:
        print("\n" + "="*50)
        print("✅ TEXTO EXTRAÍDO (MARKDOWN):")
        print("="*50)
        print(texto_bruto)
        print("="*50)

        # Extração de Dados
        print("\n" + "="*50)
        print("🔍 EXTRAÇÃO DE DADOS PESSOAIS INICIADA...")
        print("="*50)
        
        dados_extraidos = extrair_dados_pessoais(texto_bruto)
        
        for chave, valor in dados_extraidos.items():
            print(f"| {chave.upper():<18}: {valor if valor else 'NÃO ENCONTRADO'}")
        
        print("="*50)
        
    else:
        print("Não foi possível extrair o texto do documento.")

if __name__ == "__main__":
    main()