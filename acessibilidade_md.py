import re
from pathlib import Path

# Caminho do arquivo markdown de entrada
entrada  = input("ENTRE COM O ARQUIVO .md:")
arquivo_md = Path(entrada)

# Texto alternativo para imagens
texto_imagem = "[Descrição: aqui havia uma imagem ou logotipo]"

# Leitura do arquivo original
conteudo = arquivo_md.read_text(encoding="utf-8")

# Se (existir "##") então Substituir títulos por versões mais descritivas
def substituir_titulos(md):
    def marca_titulo(match):
        hashes = match.group(1)
        titulo = match.group(2).strip()
        nivel = len(hashes)
        return f"\n[Início do título nível {nivel}: {titulo}]\n"
    return re.sub(r'^(#{1,6})\s*(.+)$', marca_titulo, md, flags=re.MULTILINE)

# Substituir marcações de imagem <!-- image --> por texto alternativo
def substituir_imagens(md):
    return re.sub(r'<!--\s*image\s*-->', texto_imagem, md, flags=re.IGNORECASE)

# 3 Aplicar transformações
md_modificado = substituir_titulos(conteudo)
md_modificado = substituir_imagens(md_modificado)

# Salvar o resultado acessível
arquivo_saida = arquivo_md.with_name(arquivo_md.stem + "_acessivel.md")
arquivo_saida.write_text(md_modificado, encoding="utf-8")

print(f"Arquivo acessível criado:\n{arquivo_saida}")
