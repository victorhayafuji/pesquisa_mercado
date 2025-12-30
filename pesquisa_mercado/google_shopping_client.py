"""Cliente de Google Shopping.

Este arquivo era originalmente acoplado à SearchAPI. Para permitir um teste com
SerpApi (sem quebrar o restante do seu pipeline), a função principal agora
suporta 2 provedores:

- SearchAPI (engine=google_shopping, endpoint searchapi.io)
- SerpApi  (engine=google_shopping, endpoint serpapi.com)

O objetivo é manter o retorno como JSON bruto do provedor e continuar usando o
processador_resultados.py para padronizar em CSV.
"""

from typing import Dict, Optional, Any

import requests

from config import (
    GL_DEFAULT,
    HL_DEFAULT,
    LOCATION_DEFAULT,
    RESULTADOS_POR_PAGINA,
    SERPAPI_RESULTADOS_POR_PAGINA,
    SERPAPI_GOOGLE_DOMAIN,
    SERPAPI_LOCATION_DEFAULT,
    SERPAPI_ENGINE_PRIMARY,
    SERPAPI_ENGINE_FALLBACK,
    SERPAPI_RETRY_EMPTY,
    TIMEOUT_REQUEST,
    SEARCHAPI_API_KEY,
    SERPAPI_API_KEY,
    GOOGLE_SHOPPING_PROVIDER,
)


class GoogleShoppingError(Exception):
    """Erro genérico ao consultar Google Shopping via provedor de SERP."""
    pass


def _parse_price_to_float(value: Any) -> Optional[float]:
    """Converte preço em string para float.

    Objetivo: manter o schema final do CSV intacto, mas reduzir casos em que a
    SerpApi devolve apenas `price` (texto) sem `extracted_price` (número) e o
    pipeline acaba descartando o item.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Mantém apenas dígitos, ponto e vírgula
    cleaned = "".join(ch for ch in s if ch.isdigit() or ch in {".", ","})
    if not cleaned:
        return None

    # Caso clássico BR: 1.234,56 -> 1234.56
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    # Caso: 123,45 -> 123.45
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _get_provider(provider: Optional[str]) -> str:
    prov = (provider or GOOGLE_SHOPPING_PROVIDER or "searchapi").strip().lower()
    if prov in {"searchapi", "search_api", "search-api"}:
        return "searchapi"
    if prov in {"serpapi", "serp_api", "serp-api"}:
        return "serpapi"
    raise GoogleShoppingError(
        f"Provedor inválido: '{provider}'. Use 'searchapi' ou 'serpapi'."
    )


def _normalizar_resposta_serpapi_para_searchapi(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza a resposta da SerpApi para o *mesmo formato de entrada* esperado
    pelo processador_resultados.py (SearchAPI), SEM alterar os campos finais do CSV.

    Estratégia:
    - Garantir que exista o bloco data["shopping_results"].
    - Mapear 'source' -> 'seller' (SearchAPI usa 'seller').
    - Mapear preços antigos para chaves compatíveis ('extracted_original_price' / 'original_price'),
      pois o processador já faz fallback para 'extracted_old_price' e 'old_price'.
    - Preservar os demais campos como vierem, apenas adicionando chaves quando necessário.
    """
    if not isinstance(data, dict):
        return data  # type: ignore[return-value]

    # SerpApi costuma trazer "shopping_results". Em alguns casos, pode existir "inline_shopping_results".
    raw_results = data.get("shopping_results") or data.get("inline_shopping_results") or []

    resultados_norm = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        novo = dict(item)

        # 'seller' é o campo usado pelo seu pipeline; em SerpApi o nome típico é 'source'
        if not novo.get("seller"):
            novo["seller"] = novo.get("source")

        # Preço: se a SerpApi trouxer apenas `price` (texto) e não preencher
        # `extracted_price` (número), fazemos o parse aqui para manter o
        # comportamento do pipeline.
        if novo.get("extracted_price") is None and novo.get("price") is not None:
            parsed = _parse_price_to_float(novo.get("price"))
            if parsed is not None:
                novo["extracted_price"] = parsed

        if novo.get("extracted_old_price") is None and novo.get("old_price") is not None:
            parsed_old = _parse_price_to_float(novo.get("old_price"))
            if parsed_old is not None:
                novo["extracted_old_price"] = parsed_old

        # Preço original: garantimos chaves compatíveis com o parser atual
        if novo.get("extracted_original_price") is None and novo.get("extracted_old_price") is not None:
            novo["extracted_original_price"] = novo.get("extracted_old_price")

        if novo.get("original_price") is None and novo.get("old_price") is not None:
            novo["original_price"] = novo.get("old_price")

        # Tag promocional: o processador lê item.get("tag")
        if not novo.get("tag") and novo.get("badge"):
            badge = novo.get("badge")
            if isinstance(badge, str):
                novo["tag"] = badge
            elif isinstance(badge, dict):
                novo["tag"] = badge.get("text") or badge.get("label") or badge.get("name")

        # Links: o processador tenta product_link e depois link
        if not novo.get("product_link") and novo.get("link"):
            novo["product_link"] = novo.get("link")

        resultados_norm.append(novo)

    data_norm = dict(data)
    data_norm["shopping_results"] = resultados_norm
    return data_norm


def _buscar_via_searchapi(
    *,
    q: str,
    page: int,
    gl: str,
    hl: str,
    location: Optional[str],
    sort_by: Optional[str],
    condition: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    is_on_sale: Optional[bool],
    is_small_business: Optional[bool],
    is_free_delivery: Optional[bool],
    shoprs: Optional[str],
    include_favicon: bool,
    include_base_images: bool,
) -> Dict[str, Any]:
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

    if location:
        params["location"] = location

    if sort_by:
        params["sort_by"] = sort_by
    if condition:
        params["condition"] = condition

    if price_min is not None:
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
        params["shoprs"] = shoprs

    if include_favicon:
        params["include_favicon"] = "true"
    if include_base_images:
        params["include_base_images"] = "true"

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT_REQUEST)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise GoogleShoppingError(f"Erro de rede ao consultar SearchAPI: {e}") from e

    try:
        data = resp.json()
    except ValueError as e:
        raise GoogleShoppingError("Resposta da API não é um JSON válido.") from e

    if isinstance(data, dict) and "error" in data:
        mensagem = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        raise GoogleShoppingError(f"Erro retornado pela SearchAPI: {mensagem}")

    return data


def _buscar_via_serpapi(
    *,
    q: str,
    page: int,
    gl: str,
    hl: str,
    location: Optional[str],
    sort_by: Optional[str],
    # condition não possui equivalente direto documentado; mantemos na assinatura por compat.
    condition: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    is_on_sale: Optional[bool],
    is_small_business: Optional[bool],
    is_free_delivery: Optional[bool],
    shoprs: Optional[str],
) -> Dict[str, Any]:
    if not SERPAPI_API_KEY:
        raise GoogleShoppingError(
            "Variável de ambiente SERPAPI_API_KEY não definida. "
            "Configure sua chave da SerpApi no .env ou no ambiente."
        )

    # endpoint padrão (JSON)
    url = "https://serpapi.com/search.json"

    # A SerpApi orienta paginação via 'start' (offset). Ela costuma entregar ~60 itens por página.
    start = max(0, (max(page, 1) - 1) * SERPAPI_RESULTADOS_POR_PAGINA)

    # Base de parâmetros. Engine é aplicado na hora da requisição para
    # permitir fallback (google_shopping_light) quando vier "Fully empty".
    params_base: Dict[str, Any] = {
        "api_key": SERPAPI_API_KEY,
        "q": q,
        "gl": gl,
        "hl": hl,
        "start": start,
        "google_domain": SERPAPI_GOOGLE_DOMAIN,
    }

    # SerpApi aceita location e recomenda nível cidade.
    # Mantemos LOCATION_DEFAULT="Brazil" para SearchAPI, mas para SerpApi
    # usamos SERPAPI_LOCATION_DEFAULT quando vier vazio ou genérico.
    loc = (location or "").strip()
    if not loc or loc.lower() == "brazil":
        loc = SERPAPI_LOCATION_DEFAULT
    if loc:
        params_base["location"] = loc

    # sort_by na SerpApi é documentado como 1 (low->high) e 2 (high->low).
    if sort_by:
        sort_by_norm = str(sort_by).strip().lower()
        if sort_by_norm in {"1", "low", "price_low_to_high", "price: low to high"}:
            params_base["sort_by"] = 1
        elif sort_by_norm in {"2", "high", "price_high_to_low", "price: high to low"}:
            params_base["sort_by"] = 2

    # condition: sem equivalente direto (normalmente embutido no token shoprs). Mantemos sem ação.
    _ = condition

    if price_min is not None:
        params_base["min_price"] = price_min
    if price_max is not None:
        params_base["max_price"] = price_max

    if is_on_sale is not None:
        params_base["on_sale"] = bool(is_on_sale)
    if is_small_business is not None:
        params_base["small_business"] = bool(is_small_business)
    if is_free_delivery is not None:
        params_base["free_shipping"] = bool(is_free_delivery)

    if shoprs:
        params_base["shoprs"] = shoprs

    def _serpapi_is_empty(d: Dict[str, Any]) -> bool:
        info = d.get("search_information")
        if isinstance(info, dict):
            state = str(info.get("shopping_results_state") or "").strip().lower()
            if state in {"fully empty", "empty"}:
                return True
        # Sem resultados nos blocos usuais
        if not d.get("shopping_results") and not d.get("inline_shopping_results"):
            # Alguns casos vêm com erro textual, mas equivalem a "sem resultado"
            err = d.get("error")
            if isinstance(err, str) and "returned any results" in err.lower():
                return True
            # Se não há erro, ainda pode ser vazio real
            return True
        return False

    def _request(engine: str) -> Dict[str, Any]:
        params = dict(params_base)
        params["engine"] = engine

        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_REQUEST)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise GoogleShoppingError(f"Erro de rede ao consultar SerpApi: {e}") from e

        try:
            data = resp.json()
        except ValueError as e:
            raise GoogleShoppingError("Resposta da API não é um JSON válido.") from e

        if not isinstance(data, dict):
            raise GoogleShoppingError("Resposta inesperada da SerpApi (não é dict).")

        # SerpApi pode devolver erro em string, mesmo com status 200.
        # Caso clássico: "Google hasn't returned any results for this query.".
        # Tratamos como retorno vazio (não como exceção fatal), pois o pipeline
        # do SearchAPI costuma lidar com listas vazias.
        if "error" in data and isinstance(data.get("error"), str):
            msg = str(data.get("error"))
            if "returned any results" not in msg.lower():
                raise GoogleShoppingError(f"Erro retornado pela SerpApi: {msg}")

        # Se vier metadata de status, tratamos falhas explicitamente.
        meta = data.get("search_metadata")
        if isinstance(meta, dict):
            status = str(meta.get("status") or "").strip().lower()
            if status and status not in {"success"}:
                raise GoogleShoppingError(f"SerpApi retornou status '{meta.get('status')}'.")

        return data

    # Estratégia tradicional:
    # 1) engine principal
    # 2) retry quando vier vazio
    # 3) engine fallback (light)
    engines = [SERPAPI_ENGINE_PRIMARY]
    if SERPAPI_ENGINE_FALLBACK and SERPAPI_ENGINE_FALLBACK != SERPAPI_ENGINE_PRIMARY:
        engines.append(SERPAPI_ENGINE_FALLBACK)

    last_data: Dict[str, Any] = {"shopping_results": []}
    for engine in engines:
        for _attempt in range(max(0, SERPAPI_RETRY_EMPTY) + 1):
            data = _request(engine)
            last_data = data
            if not _serpapi_is_empty(data):
                return _normalizar_resposta_serpapi_para_searchapi(data)

    # Se ainda assim vier vazio, devolvemos estrutura vazia (sem quebrar o pipeline)
    return _normalizar_resposta_serpapi_para_searchapi(last_data)


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
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Consulta Google Shopping via provedor configurável.

    - SearchAPI: https://www.searchapi.io/docs/google-shopping
    - SerpApi:  https://serpapi.com/google-shopping-api

    Retorna o JSON bruto da API selecionada.
    """

    prov = _get_provider(provider)

    if prov == "searchapi":
        return _buscar_via_searchapi(
            q=q,
            page=page,
            gl=gl,
            hl=hl,
            location=location,
            sort_by=sort_by,
            condition=condition,
            price_min=price_min,
            price_max=price_max,
            is_on_sale=is_on_sale,
            is_small_business=is_small_business,
            is_free_delivery=is_free_delivery,
            shoprs=shoprs,
            include_favicon=include_favicon,
            include_base_images=include_base_images,
        )

    # prov == "serpapi"
    return _buscar_via_serpapi(
        q=q,
        page=page,
        gl=gl,
        hl=hl,
        location=location,
        sort_by=sort_by,
        condition=condition,
        price_min=price_min,
        price_max=price_max,
        is_on_sale=is_on_sale,
        is_small_business=is_small_business,
        is_free_delivery=is_free_delivery,
        shoprs=shoprs,
    )