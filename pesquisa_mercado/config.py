# config.py

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env (incluindo SEARCHAPI_API_KEY)
load_dotenv()

# Diretório base = pasta onde está este arquivo
BASE_DIR = Path(__file__).resolve().parent

# Diretório onde os arquivos de saída (CSV/Excel) serão salvos
OUTPUT_DIR = BASE_DIR / "outputs"

# =========================
# Parâmetros padrão da SearchAPI / Google Shopping
# =========================

# País (geolocalização) - padrão Brasil
GL_DEFAULT = "br"

# Idioma da interface do Google
HL_DEFAULT = "pt"

# Localização textual usada pela SearchAPI
LOCATION_DEFAULT = "Brazil"

# Quantidade de resultados por página no Google Shopping via SearchAPI
# A SearchAPI costuma trabalhar com blocos de até ~40 itens por página.
RESULTADOS_POR_PAGINA = 40

# Chave da API da SearchAPI (via .env)
SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY")

# Timeout (em segundos) para cada requisição HTTP à SearchAPI.
# Se quiser ajustar sem mexer no código, defina a variável de ambiente
# SEARCHAPI_TIMEOUT. Caso contrário, usa o valor padrão abaixo.
try:
    TIMEOUT_REQUEST = int(os.getenv("SEARCHAPI_TIMEOUT", "45"))
except ValueError:
    TIMEOUT_REQUEST = 1000

# =========================
# Controles de coleta
# =========================

# Número padrão de páginas a buscar no Google Shopping
# Ex.: 4 páginas x 40 resultados ≈ até 160 itens (se o Google devolver tudo)
N_PAGINAS_DEFAULT = 4

# Pausa (em segundos) entre requisições à API.
# Use 0 para não pausar.
PAUSA_ENTRE_REQUISICOES = 0
