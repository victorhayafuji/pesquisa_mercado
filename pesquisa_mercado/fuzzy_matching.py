# fuzzy_matching.py
"""Utilitários de Fuzzy Matching para normalizar marcas de produtos."""

from typing import Iterable, List, Optional, Tuple
import csv

try:
    # RapidFuzz: fuzzy matching moderno e performático
    # pip install rapidfuzz
    from rapidfuzz import process, fuzz, utils as fuzz_utils  # type: ignore[import]
except ImportError:
    process = None  # type: ignore[assignment]
    fuzz = None  # type: ignore[assignment]
    fuzz_utils = None  # type: ignore[assignment]

from config import BASE_DIR  # já existente no projeto


# Diretório dos arquivos referenciais
REFERENCIAIS_DIR = BASE_DIR / "referenciais"
MARCAS_CANONICAS_CSV = REFERENCIAIS_DIR / "marcas_canonicas.csv"
MARCAS_NOVAS_CSV = REFERENCIAIS_DIR / "marcas_novas_encontradas.csv"


def carregar_lista_canonica_marcas(
    marcas_base: Optional[Iterable[str]] = None,
) -> List[str]:
    """Carrega lista de marcas canônicas a partir de:
    - lista base (MARCAS_KNOWN)
    - CSV marcas_canonicas.csv, se existir
    """
    REFERENCIAIS_DIR.mkdir(parents=True, exist_ok=True)

    canonicas: set[str] = set()

    if marcas_base:
        for m in marcas_base:
            m_limpo = (m or "").strip()
            if m_limpo:
                canonicas.add(m_limpo)

    if MARCAS_CANONICAS_CSV.exists():
        with MARCAS_CANONICAS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None:
                f.seek(0)
                reader = csv.DictReader(f)
            for row in reader:
                m = (row.get("marca") or "").strip()
                if m:
                    canonicas.add(m)

    return sorted(canonicas)


# Cache em memória para evitar registrar várias vezes a mesma marca na mesma execução
_MARCAS_NOVAS_SESSAO: set[str] = set()


def registrar_marca_nova(raw: Optional[str]) -> None:
    """Registra marcas não reconhecidas para futura revisão manual."""
    if not raw:
        return

    marca_limpa = (raw or "").strip()
    if not marca_limpa:
        return

    chave = marca_limpa.lower()
    if chave in _MARCAS_NOVAS_SESSAO:
        return

    _MARCAS_NOVAS_SESSAO.add(chave)

    REFERENCIAIS_DIR.mkdir(parents=True, exist_ok=True)

    escrever_header = not MARCAS_NOVAS_CSV.exists()
    with MARCAS_NOVAS_CSV.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        if escrever_header:
            writer.writerow(["marca_raw"])
        writer.writerow([marca_limpa])


def fuzzy_melhor_marca(
    candidato: Optional[str],
    lista_canonicas: Iterable[str],
    threshold: float = 88.0,
) -> Optional[str]:
    """Retorna a melhor marca canônica via fuzzy matching, se score >= threshold.

    Se RapidFuzz não estiver instalado, retorna sempre None.
    """
    if process is None or fuzz is None or fuzz_utils is None:
        return None

    if not candidato:
        return None

    candidato_norm = candidato.strip()
    if not candidato_norm:
        return None

    choices = [c for c in lista_canonicas if (c or "").strip()]
    if not choices:
        return None

    resultado: Optional[Tuple[str, float, int]] = process.extractOne(
        candidato_norm,
        choices,
        scorer=fuzz.WRatio,
        processor=fuzz_utils.default_process,
    )

    if not resultado:
        return None

    melhor_marca, score, _ = resultado
    if score >= threshold:
        return melhor_marca

    return None
