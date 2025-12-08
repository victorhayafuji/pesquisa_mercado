# config.py

from pathlib import Path

# Diretório base = pasta onde está este arquivo
BASE_DIR = Path(__file__).resolve().parent

# Diretório onde os arquivos de saída (CSV/Excel) serão salvos
OUTPUT_DIR = BASE_DIR / "outputs"

# Número padrão de páginas a buscar no Google Shopping
N_PAGINAS_DEFAULT = 4

# Parâmetros padrão de localização / idioma da busca
GL_DEFAULT = "br"          # Country (Brasil)
HL_DEFAULT = "pt"          # Idioma da interface (Português)
LOCATION_DEFAULT = "Brazil"

# Número de resultados por página (SearchAPI suporta num; exemplo oficial usa 60)
RESULTADOS_POR_PAGINA = 60