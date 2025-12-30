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

# =========================
# Provedor de coleta (SearchAPI x SerpApi)
# =========================

# Define qual backend usar no google_shopping_client.py.
# Valores esperados: "searchapi" (padrão) ou "serpapi".
GOOGLE_SHOPPING_PROVIDER = os.getenv("GOOGLE_SHOPPING_PROVIDER", "searchapi").strip().lower()

# Chave da API da SerpApi (via .env)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# =========================
# Ajustes específicos SerpApi (para reduzir casos de "0 resultados")
# =========================

# Domínio do Google usado pela SerpApi.
# O default da SerpApi costuma ser google.com, mas para o Brasil
# o comportamento tende a ficar mais consistente com google.com.br.
SERPAPI_GOOGLE_DOMAIN = os.getenv("SERPAPI_GOOGLE_DOMAIN", "google.com.br").strip()

# Location recomendado (nível cidade) para simular melhor um usuário real.
# Mantemos LOCATION_DEFAULT="Brazil" para SearchAPI, mas para SerpApi usamos
# este fallback quando location vier vazio ou genérico.
SERPAPI_LOCATION_DEFAULT = os.getenv(
    "SERPAPI_LOCATION_DEFAULT",
    "Sao Paulo, State of Sao Paulo, Brazil",
).strip()

# Engine principal e fallback. O fallback "google_shopping_light" costuma
# ser útil quando o layout do Shopping está instável.
SERPAPI_ENGINE_PRIMARY = os.getenv("SERPAPI_ENGINE_PRIMARY", "google_shopping").strip()
SERPAPI_ENGINE_FALLBACK = os.getenv("SERPAPI_ENGINE_FALLBACK", "google_shopping_light").strip()

# Quantas tentativas extras (retry) fazer quando o retorno vier "Fully empty".
try:
    SERPAPI_RETRY_EMPTY = int(os.getenv("SERPAPI_RETRY_EMPTY", "1"))
except ValueError:
    SERPAPI_RETRY_EMPTY = 0

# A SerpApi documenta a paginação pelo parâmetro "start" em saltos típicos de 60.
# Para evitar duplicidade / buracos entre páginas, usamos 60 como default.
try:
    SERPAPI_RESULTADOS_POR_PAGINA = int(os.getenv("SERPAPI_RESULTADOS_POR_PAGINA", "60"))
except ValueError:
    SERPAPI_RESULTADOS_POR_PAGINA = 60

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
N_PAGINAS_DEFAULT = 2

# Pausa (em segundos) entre requisições à API.
# Use 0 para não pausar.
PAUSA_ENTRE_REQUISICOES = 0
