# processador_resultados.py

from typing import List, Dict, Optional
import re

# Tentamos usar RapidFuzz para fuzzy matching em brand_api; se não estiver instalado, seguimos sem fuzzy.
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz  # type: ignore[import]
except ImportError:  # pragma: no cover - ambiente sem rapidfuzz
    rf_process = None  # type: ignore[assignment]
    rf_fuzz = None     # type: ignore[assignment]

# Importa a lista de marcas conhecidas do módulo de referenciais
from referenciais.marcas_conhecidas import MARCAS_KNOWN


def _normalizar_marca_brand_api(brand_api: Optional[str]) -> Optional[str]:
    """
    Normaliza a marca vinda do campo estruturado 'brand' da SearchAPI.

    Regras:
    - Se RapidFuzz não estiver disponível, apenas capitaliza (title case).
    - Se fuzzy estiver disponível e o score contra MARCAS_KNOWN for >= 90,
      retorna a marca canônica da lista.
    - Caso contrário, retorna brand_api apenas capitalizada.
    """
    if not brand_api:
        return None

    marca_bruta = brand_api.strip()
    if not marca_bruta:
        return None

    # Sem RapidFuzz: só padroniza a capitalização
    if rf_process is None or rf_fuzz is None:
        return marca_bruta.title()

    # Choices canônicos
    choices = [m.title() for m in MARCAS_KNOWN]

    resultado = rf_process.extractOne(
        marca_bruta,
        choices,
        scorer=rf_fuzz.WRatio,
    )

    if not resultado:
        return marca_bruta.title()

    marca_canonica, score, _ = resultado

    if score >= 90.0:
        return marca_canonica

    # Score baixo: não forçamos, usamos a marca como veio
    return marca_bruta.title()


def inferir_marca(
    titulo: Optional[str],
    seller: Optional[str] = None,   # mantido só por compatibilidade da assinatura
    brand_api: Optional[str] = None,
) -> Optional[str]:
    """
    Define a marca com a seguinte prioridade:

    1) Se brand_api vier preenchida:
       - tenta normalizar via fuzzy contra MARCAS_KNOWN;
       - se não der match forte, usa brand_api capitalizada.

    2) Se não houver brand_api, procura marcas conhecidas no título:
       - Para a marca 'Ou', usamos regra específica:
         * se houver token 'Ou' ou 'OU' (O maiúsculo) no título, retornamos 'Ou';
         * 'ou' minúsculo é tratado como conjunção, não como marca.
       - Para as demais marcas, usamos substring no título em minúsculas.

    3) Se nada encontrado, retorna None (não usamos mais seller como fallback).
    """
    # 1) Fonte estruturada tem prioridade
    if brand_api:
        return _normalizar_marca_brand_api(brand_api)

    # 2) Sem brand_api: tentamos pelo título
    if not titulo:
        return None

    # Tokens com a grafia original (para diferenciar 'Ou' de 'ou')
    tokens_originais = re.findall(r"\b\w+\b", titulo)
    tokens_lower = [t.lower() for t in tokens_originais]  # mantido se quiser usar depois

    # Regra específica para marca 'Ou':
    # - se aparecer 'Ou' ou 'OU' como palavra isolada, consideramos marca
    if "Ou" in tokens_originais or "OU" in tokens_originais:
        return "Ou"

    # Agora buscamos as demais marcas por substring em minúsculas
    nome_lower = titulo.lower()

    for marca in MARCAS_KNOWN:
        m = marca.lower()

        # Já tratamos 'Ou' acima; aqui pulamos para evitar conflito
        if m == "ou":
            continue

        if m in nome_lower:
            return marca.title()

    # 3) Nada encontrado
    return None


def extrair_resultados_basicos(
    resposta_json: Dict,
    palavra_chave: str,
    fonte: str = "google_shopping",
) -> List[Dict]:
    """
    Converte o JSON bruto da SearchAPI em uma lista de dicionários
    no formato:
    {
        "fonte": "google_shopping",
        "palavra_chave": "...",
        "produto": "...",
        "preco": 123.45,
        "preco_original": 199.90,
        "tag_promocao": "14% OFF",
        "seller": "...",
        "marca": "...",
        "product_id": "...",
        "posicao": 1,
        "rating": 4.7,
        "reviews": 2853,
        "delivery": "...",
        "condicao_anuncio": "...",
        "link": "https://..."
    }

    Removemos o campo de categoria/tipo de produto, pois a segmentação
    virá da própria palavra-chave usada na pesquisa.
    """
    data = resposta_json or {}

    resultados: List[Dict] = []

    shopping_results = (
        data.get("shopping_results")
        or data.get("popular_products")
        or []
    )

    for item in shopping_results:
        titulo = item.get("title")
        preco = item.get("extracted_price")
        seller = item.get("seller")
        link = item.get("product_link") or item.get("link")
        brand_api = item.get("brand")

        # Ignora itens sem título ou sem preço numérico
        if not titulo or preco is None:
            continue

        marca = inferir_marca(titulo, seller, brand_api)

        # Campos adicionais para análise mais rica
        position = item.get("position")
        product_id = item.get("product_id")
        rating = item.get("rating")
        reviews = item.get("reviews")
        delivery = item.get("delivery")
        tag_promocao = item.get("tag")  # ex.: "14% OFF"
        preco_original = (
            item.get("extracted_original_price")
            or item.get("original_price")
            or item.get("extracted_old_price")
            or item.get("old_price")
        )
        condicao_anuncio = (
            item.get("second_hand_condition")
            or item.get("durability")
        )

        resultados.append(
            {
                "fonte": fonte,
                "palavra_chave": palavra_chave,
                "produto": titulo,
                "preco": preco,
                "preco_original": preco_original,
                "tag_promocao": tag_promocao,
                "seller": seller,
                "marca": marca,
                "product_id": product_id,
                "posicao": position,
                "rating": rating,
                "reviews": reviews,
                "delivery": delivery,
                "condicao_anuncio": condicao_anuncio,
                "link": link,
            }
        )

    return resultados
