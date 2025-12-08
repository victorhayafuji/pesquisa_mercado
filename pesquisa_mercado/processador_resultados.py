# processador_resultados.py

from typing import List, Dict, Any, Optional


# Lista de marcas conhecidas – heurístico NOSSO (não é dado da API)
MARCAS_KNOWN = [
    "tramontina",
    "brinox",
    "euro home",
    "flash limp",
    "flashlimp",
    "ou",
    "panelux",
    "mta",
    "le creuset",
    "ceraflame",
    "sanremo",
    "condor",
    "bettanin",
    "nobrecar",
    "multilaser",
    "ou home",
    "euro",
    "mimo",
    "colorstone"
]


def _inferir_marca(title: str, seller: Optional[str]) -> Optional[str]:
    """
    Heurística simples de marca, baseada em título + seller.

    Importante: este campo NÃO é nativo da SearchAPI.
    É uma classificação nossa, baseada em regras.
    """
    texto = (title or "") + " " + (seller or "")
    texto_norm = texto.lower()

    for marca in MARCAS_KNOWN:
        if marca in texto_norm:
            # Devolvemos com capitalização “bonitinha”
            return marca.title()

    return None


def _classificar_tipo_produto(title: str, palavra_chave: str) -> str:
    """
    Classificação simplificada do tipo de produto,
    baseada em palavras-chave no título + palavra pesquisada.

    Também é um campo derivado, não nativo da API.
    """
    texto = f"{title} {palavra_chave}".lower()

    if "jogo" in texto and "panela" in texto:
        return "jogo_panelas"
    if "caçarola" in texto or "cacarola" in texto:
        return "cacarola"
    if "frigideira" in texto:
        return "frigideira"
    if "mop" in texto and "spray" in texto:
        return "mop_spray"
    if "mop" in texto:
        return "mop_outros"

    return "outros"


def _extrair_lista_resultados(resposta_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Junta shopping_results e popular_products em uma única lista,
    mantendo a informação de qual seção originou o item.
    """
    resultados: List[Dict[str, Any]] = []

    for secao in ("shopping_results", "popular_products"):
        for item in resposta_json.get(secao, []) or []:
            if not isinstance(item, dict):
                continue
            # Guardamos a origem para rastreabilidade
            item_copia = dict(item)
            item_copia["_secao_origem"] = secao
            resultados.append(item_copia)

    return resultados


def extrair_resultados_basicos(
    resposta_json: Dict[str, Any],
    palavra_chave: str,
) -> List[Dict[str, Any]]:
    """
    Extrai os campos mais relevantes dos resultados da SearchAPI
    (alinhado ao schema ShoppingResult) e adiciona colunas de negócio.

    Campos nativos aproveitados:
    - position, product_id, prds
    - title, product_link
    - seller, offers, extracted_offers, offers_link
    - price, extracted_price
    - original_price, extracted_original_price
    - rating, reviews, tag
    - delivery, delivery_return
    - condition
    - thumbnail, thumbnail_video, product_token

    Campos DERIVADOS:
    - fonte, secao_origem, palavra_chave, tipo_produto, marca
    """

    resultados_processados: List[Dict[str, Any]] = []

    itens = _extrair_lista_resultados(resposta_json)

    for item in itens:
        title = item.get("title", "")
        seller = item.get("seller")

        product_id = item.get("product_id") or item.get("prds")
        position = item.get("position")

        price_display = item.get("price")
        price_num = item.get("extracted_price")

        original_price_display = item.get("original_price")
        original_price_num = item.get("extracted_original_price")

        offers_display = item.get("offers")
        offers_num = item.get("extracted_offers")

        rating = item.get("rating")
        reviews = item.get("reviews")
        tag = item.get("tag")

        delivery = item.get("delivery") or item.get("delivery_return")
        condition = item.get("condition")  # conforme o schema da SearchAPI

        product_link = item.get("product_link")
        offers_link = item.get("offers_link")

        thumbnail = item.get("thumbnail")
        thumbnail_video = item.get("thumbnail_video")
        product_token = item.get("product_token")

        # Campos de negócio (heurísticos)
        tipo_produto = _classificar_tipo_produto(title, palavra_chave)
        marca = _inferir_marca(title, seller)

        resultados_processados.append(
            {
                # Metadados de origem
                "fonte": "google_shopping",
                "secao_origem": item.get("_secao_origem"),
                "palavra_chave": palavra_chave,

                # Identificação e posição
                "posicao": position,
                "product_id": product_id,

                # Produto / lojista
                "produto": title,
                "tipo_produto": tipo_produto,
                "seller": seller,
                "marca": marca,

                # Preços
                "preco_exibicao": price_display,
                "preco": price_num,
                "preco_original_exibicao": original_price_display,
                "preco_original": original_price_num,

                # Ofertas
                "ofertas_exibicao": offers_display,
                "ofertas": offers_num,

                # Performance comercial
                "rating": rating,
                "reviews": reviews,
                "tag": tag,

                # Logística e condição
                "delivery": delivery,
                "condicao_anuncio": condition,

                # Links
                "link_produto_google": product_link,
                "link_ofertas": offers_link,

                # Mídia
                "thumbnail": thumbnail,
                "thumbnail_video": thumbnail_video,
                "product_token": product_token,
            }
        )

    return resultados_processados