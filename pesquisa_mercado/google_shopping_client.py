# google_shopping_client.py

import os
from typing import Dict, Optional, Any

import requests
from dotenv import load_dotenv

from config import GL_DEFAULT, HL_DEFAULT, LOCATION_DEFAULT, RESULTADOS_POR_PAGINA

load_dotenv()

SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY")


class GoogleShoppingError(Exception):
    """Erro genérico ao consultar Google Shopping via SearchAPI."""
    pass


def buscar_google_shopping(
    q: str,
    page: int = 1,
    gl: str = GL_DEFAULT,
    hl: str = HL_DEFAULT,
    location: Optional[str] = LOCATION_DEFAULT,
    sort_by: Optional[str] = None,          # 'price_low_to_high', 'price_high_to_low', 'rating_high_to_low'
    condition: Optional[str] = None,        # 'new' ou 'used'
    price_min: Optional[float] = None,      # ex.: 20.0
    price_max: Optional[float] = None,      # ex.: 200.0
    is_on_sale: Optional[bool] = None,      # True para somente itens em promoção (via shoprs / flags internos)
    is_small_business: Optional[bool] = None,
    is_free_delivery: Optional[bool] = None,
    shoprs: Optional[str] = None,           # token de filtro retornado em "filters"
    include_favicon: bool = False,
    include_base_images: bool = False,
) -> Dict[str, Any]:
    """
    Consulta o endpoint oficial da SearchAPI para Google Shopping.

    Formato alinhado à documentação em:
    https://www.searchapi.io/docs/google-shopping

    Retorna o JSON completo da API (SearchResponse).
    """

    if not SEARCHAPI_API_KEY:
        raise GoogleShoppingError(
            "Variável de ambiente SEARCHAPI_API_KEY não definida. "
            "Configure sua chave da SearchAPI no .env ou no ambiente."
        )

    url = "https://www.searchapi.io/api/v1/search"

    params: Dict[str, Any] = {
        "api_key": SEARCHAPI_API_KEY,
        "engine": "google_shopping",
        "q": q,
        "page": page,
        "gl": gl,
        "hl": hl,
        "num": RESULTADOS_POR_PAGINA,
    }

    # Campos opcionais de localização
    if location:
        params["location"] = location

    # Filtros opcionais (seguindo parâmetros da doc)
    if sort_by:
        params["sort_by"] = sort_by  # 'price_low_to_high', 'price_high_to_low', 'rating_high_to_low'
    if condition:
        params["condition"] = condition  # 'new' ou 'used'

    if price_min is not None:
        # Na doc, price_min/price_max são strings numéricas — usamos float convertido para string
        params["price_min"] = str(price_min)
    if price_max is not None:
        params["price_max"] = str(price_max)

    if is_on_sale is not None:
        params["is_on_sale"] = str(is_on_sale).lower()
    if is_small_business is not None:
        params["is_small_business"] = str(is_small_business).lower()
    if is_free_delivery is not None:
        params["is_free_delivery"] = str(is_free_delivery).lower()

    if shoprs:
        # shoprs tem prioridade sobre alguns filtros, como indicado na doc
        params["shoprs"] = shoprs

    if include_favicon:
        params["include_favicon"] = "true"
    if include_base_images:
        params["include_base_images"] = "true"

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise GoogleShoppingError(f"Erro de rede ao consultar SearchAPI: {e}") from e

    try:
        data = resp.json()
    except ValueError as e:
        raise GoogleShoppingError("Resposta da API não é um JSON válido.") from e

    # Se a API retornar um bloco de erro no JSON, tratamos aqui
    if isinstance(data, dict) and "error" in data:
        mensagem = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        raise GoogleShoppingError(f"Erro retornado pela SearchAPI: {mensagem}")

    return data