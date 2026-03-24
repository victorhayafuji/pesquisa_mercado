import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =========================
# Config (executivo)
# =========================
st.set_page_config(page_title="Régua de Preço", layout="wide")

EXEC_CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 1.6rem; }
h1 { font-size: 1.55rem; margin-bottom: .2rem; }
h2 { font-size: 1.15rem; margin-top: 1rem; }
h3 { font-size: 1.0rem; margin-top: .8rem; }
small, .caption { color: #6b7280; }
div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stDataFrame"] { border-radius: 10px; }
.stDownloadButton button { border-radius: 10px; padding: 0.55rem 0.85rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
"""
st.markdown(EXEC_CSS, unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def parse_price_to_float(value: str) -> float:
    if value is None:
        return np.nan
    txt = str(value).strip()
    if txt == "":
        return np.nan

    txt = re.sub(r"[^\d,.\-]", "", txt)

    # BR: 1.234,56
    if txt.count(",") == 1 and txt.count(".") >= 1:
        txt = txt.replace(".", "").replace(",", ".")
    else:
        # 123,45
        if txt.count(",") == 1 and txt.count(".") == 0:
            txt = txt.replace(",", ".")
        # 1.234.567
        if txt.count(".") > 1 and txt.count(",") == 0:
            txt = txt.replace(".", "")

    try:
        return float(txt)
    except ValueError:
        return np.nan


def winsorize(x: np.ndarray, p_low: float = 1, p_high: float = 99) -> tuple[np.ndarray, dict]:
    """Winsoriza (corta) extremos em percentis. Não remove linhas; apenas limita valores para o cálculo."""
    lo, hi = np.percentile(x, [p_low, p_high])
    xw = np.clip(x, lo, hi)
    meta = {
        "metodo": f"Winsor P{int(p_low)}/P{int(p_high)}",
        "lim_inf": float(lo),
        "lim_sup": float(hi),
        "outliers": int(np.sum((x < lo) | (x > hi))),
    }
    return xw, meta

def iqr_filter(x: np.ndarray, k: float = 1.5) -> tuple[np.ndarray, dict]:
    """Remove outliers pelo critério IQR (Tukey)."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (x >= lo) & (x <= hi)
    meta = {
        "metodo": f"IQR {k}x",
        "lim_inf": float(lo),
        "lim_sup": float(hi),
        "outliers": int(np.sum(~mask)),
    }
    return x[mask], meta

def mad_filter(x: np.ndarray, z: float = 3.5) -> tuple[np.ndarray, dict]:
    """Remove outliers via MAD (Median Absolute Deviation), robusto a caudas longas."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return x, {"metodo": f"MAD z={z}", "lim_inf": np.nan, "lim_sup": np.nan, "outliers": 0}

    # robust z-score
    rz = 0.6745 * (x - med) / mad
    mask = np.abs(rz) <= z
    meta = {"metodo": f"MAD z={z}", "lim_inf": np.nan, "lim_sup": np.nan, "outliers": int(np.sum(~mask))}
    return x[mask], meta


def df_to_csv_bytes(df: pd.DataFrame, sep=";", encoding="utf-8-sig") -> bytes:
    return df.to_csv(index=False, sep=sep).encode(encoding)

def brl(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def brl_tick(x, pos=None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s

# =========================
# Gráficos (executivo)
# =========================
def short_label(s: str, max_len: int = 18) -> str:
    s = "" if s is None else str(s)
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def plot_barh_top(series: pd.Series, title: str, xlabel: str = "Linhas", max_items: int = 10):
    s = series.head(max_items)
    labels = [short_label(x, 26) for x in s.index.astype(str)]
    values = s.values

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(values))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)

    pad = (max(values) * 0.01) if len(values) else 1
    for i, v in enumerate(values):
        ax.text(v + pad, i, f"{int(v)}", va="center", fontsize=10)

    fig.tight_layout()
    return fig

def plot_pareto(series: pd.Series, base_total: int, title: str, max_items: int = 12):
    s = series.head(max_items)
    labels = [short_label(x, 18) for x in s.index.astype(str)]
    values = s.values

    cum = (s.cumsum() / base_total * 100).values
    cobertura = float(cum[-1]) if len(cum) else 0.0

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(values))

    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Linhas")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        f"{title} — Cobertura Top {len(values)}: {cobertura:.1f}%",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    ymax = max(values) if len(values) else 0
    for i, v in enumerate(values):
        ax.text(i, v + (ymax * 0.01), f"{int(v)}", ha="center", va="bottom", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(x, cum, marker="o", color="#6D28D9", linewidth=2)
    ax2.set_ylabel("% Acumulado")
    ax2.set_ylim(0, 105)

    for i, p in enumerate(cum):
        if i in (0, 1, 2, len(cum) - 1):
            ax2.text(i, p + 2, f"{p:.0f}%", ha="center", fontsize=9, color="#6D28D9")

    fig.tight_layout()
    return fig

# =========================
# Limpeza (somente remoção)
# =========================
def clean_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    has_kw = "palavra_chave" in df.columns
    has_prod = "produto" in df.columns
    has_link = "link" in df.columns
    has_seller = "seller" in df.columns
    has_price = "preco" in df.columns

    if not has_prod:
        return df.copy(), df.iloc[0:0].copy(), {
            "linhas_original": int(len(df)),
            "linhas_tratado": int(len(df)),
            "linhas_removidos": 0,
            "top_motivos": {},
            "observacao": "Sem coluna 'produto': limpeza por tema não aplicada."
        }

    PAT_VINIL = r"\bvinil\b"
    PAT_PAPEL_PAREDE = r"papel\s+de\s+parede"
    PAT_CONTACT = r"\bcontact\b"
    PAT_IMPRESS = r"\bimpress(?:ao|ão|)\b|impress"
    PAT_PLOTTER = r"\bplotter\b"
    PAT_LAMINACAO = r"lamina(?:ç|c)ao|laminação|laminacao"
    OFFTOPIC_REGEX = re.compile(
        "|".join([PAT_VINIL, PAT_PAPEL_PAREDE, PAT_CONTACT, PAT_IMPRESS, PAT_PLOTTER, PAT_LAMINACAO]),
        re.IGNORECASE
    )

    def norm(x): return str(x).strip().lower() if x is not None else ""

    remove_mask = pd.Series(False, index=df.index)
    motivos = {}

    def add_reason(i, r):
        motivos.setdefault(i, []).append(r)

    # 5.1 Fora do tema
    if has_kw:
        for i, row in df.iterrows():
            kw = norm(row.get("palavra_chave", ""))
            prod = norm(row.get("produto", ""))

            if kw == "":
                continue

            if kw == "rolos adesivos":
                keep_terms = ["tira pelos", "tira-pelos", "pelos", "roupa", "pet", "refil"]
                if any(t in prod for t in keep_terms):
                    continue
                if OFFTOPIC_REGEX.search(prod):
                    remove_mask.at[i] = True
                    add_reason(i, "Fora do tema: rolos adesivos")

            elif kw == "panos multiuso":
                if OFFTOPIC_REGEX.search(prod) or any(t in prod for t in ["decorativ", "decoração"]):
                    remove_mask.at[i] = True
                    add_reason(i, "Fora do tema: panos multiuso")

            elif kw == "anti mofo":
                rel_terms = ["mofo", "umidad", "desumid", "absorv", "antimofo", "anti mofo"]
                if OFFTOPIC_REGEX.search(prod) and not any(t in prod for t in rel_terms):
                    remove_mask.at[i] = True
                    add_reason(i, "Fora do tema: anti mofo")

            elif kw == "esponjas inox":
                if any(t in prod for t in ["lavabo", "banheiro", "escova sanit", "lixeira", "suporte", "porta", "prateleira", "dispenser"]):
                    remove_mask.at[i] = True
                    add_reason(i, "Fora do tema: esponjas inox (banheiro/organização)")

            else:
                if OFFTOPIC_REGEX.search(prod):
                    remove_mask.at[i] = True
                    add_reason(i, "Fora do tema: indicador forte (vinil/papel/contact etc.)")

    # 5.2 Duplicidade
    if has_link:
        link = df["link"].astype(str)
        filled = link.str.strip() != ""
        dup_link = filled & link.duplicated(keep="first")
        for i in df.index[dup_link]:
            remove_mask.at[i] = True
            add_reason(i, "Duplicado: mesmo link")

        empty = link.str.strip() == ""
        if has_seller and has_price:
            key = df["produto"].astype(str) + "||" + df["seller"].astype(str) + "||" + df["preco"].astype(str)
            dup = empty & key.duplicated(keep="first")
            for i in df.index[dup]:
                remove_mask.at[i] = True
                add_reason(i, "Duplicado: produto+seller+preço (link vazio)")
        elif has_price:
            key = df["produto"].astype(str) + "||" + df["preco"].astype(str)
            dup = empty & key.duplicated(keep="first")
            for i in df.index[dup]:
                remove_mask.at[i] = True
                add_reason(i, "Duplicado: produto+preço (link vazio)")

    # 5.3 Preço incoerente (óbvio)
    if has_price:
        pnum = df["preco"].map(parse_price_to_float)
        prod = df["produto"].map(norm)

        small_item_terms = [
            "refil", "refill", "pano", "panos", "esponja", "esponjas", "rolo", "rolos",
            "tira pelos", "tira-pelos", "absorvedor", "anti mofo", "antimofo", "lã de aço",
            "la de aco", "palha de aço", "palha de aco"
        ]
        bulk_terms = ["kit", "caixa", "fardo", "atacado", "unid", "unidades", "pacote", "c/"]

        incoerente = (pnum >= 10000) & prod.map(lambda t: any(s in t for s in small_item_terms)) & (~prod.map(lambda t: any(s in t for s in bulk_terms)))
        for i in df.index[incoerente.fillna(False)]:
            if not remove_mask.at[i]:
                remove_mask.at[i] = True
                add_reason(i, "Preço incoerente (óbvio)")

    df_tratado = df.loc[~remove_mask, df.columns].copy()
    df_removidos = df.loc[remove_mask, df.columns].copy()

    flat = [r for rs in motivos.values() for r in rs]
    top_motivos = pd.Series(flat).value_counts().to_dict() if flat else {}

    meta = {
        "linhas_original": int(len(df)),
        "linhas_tratado": int(len(df_tratado)),
        "linhas_removidos": int(len(df_removidos)),
        "top_motivos": top_motivos,
        "observacao": ""
    }
    return df_tratado, df_removidos, meta

# =========================
# Régua (em cima do tratado)
# =========================
def build_regua(df_tratado: pd.DataFrame, outlier_mode: str = "Winsor P1/P99") -> tuple[pd.DataFrame, dict]:
    """
    Calcula a régua de preço em cima da BASE TRATADA.

    Governança:
    - NÃO altera o df_tratado.
    - Tratamento de outliers é aplicado SOMENTE no cálculo da régua (p_calc).
    """

    if "preco" not in df_tratado.columns:
        raise ValueError("Coluna 'preco' não encontrada.")

    has_kw = "palavra_chave" in df_tratado.columns

    pnum = df_tratado["preco"].map(parse_price_to_float)
    valid = pnum.notna()
    df_valid = df_tratado.loc[valid].copy()
    df_valid["_preco_num"] = pnum.loc[valid].values

    if has_kw:
        grouped = df_valid.groupby(df_valid["palavra_chave"].replace("", "(vazio)"))
    else:
        grouped = [("(geral)", df_valid)]

    rows = []
    outliers_total = 0

    for key, g in grouped:
        p = g["_preco_num"].to_numpy(dtype=float)
        if len(p) == 0:
            continue

        # ------------------------------
        # Outliers (somente para cálculo)
        # ------------------------------
        p_calc = p
        meta_out = {"metodo": "Nenhum", "outliers": 0, "lim_inf": np.nan, "lim_sup": np.nan}

        # Governança: não tratar outliers se amostra pequena
        if len(p) >= 30:
            if outlier_mode == "Winsor P1/P99":
                p_calc, meta_out = winsorize(p, 1, 99)
            elif outlier_mode == "IQR 1.5x":
                p_calc, meta_out = iqr_filter(p, 1.5)
            elif outlier_mode == "IQR 3.0x":
                p_calc, meta_out = iqr_filter(p, 3.0)
            elif outlier_mode == "MAD z=3.5":
                p_calc, meta_out = mad_filter(p, 3.5)
            else:
                p_calc = p
                meta_out = {"metodo": "Nenhum", "outliers": 0, "lim_inf": np.nan, "lim_sup": np.nan}

            # Segurança: se o filtro removeu demais, recua
            if meta_out["metodo"].startswith("IQR") or meta_out["metodo"].startswith("MAD"):
                if len(p_calc) < max(10, int(0.60 * len(p))):
                    p_calc = p
                    meta_out = {"metodo": "Recuo (segurança)", "outliers": 0, "lim_inf": np.nan, "lim_sup": np.nan}

        outliers_total += int(meta_out.get("outliers", 0))

        # ------------------------------
        # Cálculo da régua (p_calc)
        # ------------------------------
        mn = float(np.min(p_calc))
        md = float(np.mean(p_calc))
        mx = float(np.max(p_calc))

        p33, p67 = (np.nan, np.nan)
        if len(p_calc) >= 2:
            p33, p67 = np.percentile(p_calc, [33, 67])

        rows.append({
            "palavra_chave": key,
            "n_precos": int(len(p)),
            "preco_min": mn,
            "preco_medio": md,
            "preco_max": mx,
            "corte_P33": float(p33) if not np.isnan(p33) else np.nan,
            "corte_P67": float(p67) if not np.isnan(p67) else np.nan,
            "faixa_economica": f"{brl(mn)} até {brl(p33)}" if not np.isnan(p33) else "",
            "faixa_media": f"{brl(p33)} até {brl(p67)}" if not np.isnan(p33) and not np.isnan(p67) else "",
            "faixa_premium": f"acima de {brl(p67)}" if not np.isnan(p67) else "",
            "amostra_pequena": "sim" if len(p) < 30 else "nao",
            "metodo_outlier": meta_out.get("metodo", "Nenhum"),
            "outliers_mitigados": int(meta_out.get("outliers", 0)),
        })

    regua = pd.DataFrame(rows).sort_values("palavra_chave")

    meta = {
        "linhas_tratado": int(len(df_tratado)),
        "precos_validos": int(valid.sum()),
        "precos_invalidos": int((~valid).sum()),
        "segmentado": bool(has_kw),
        "metodo_outlier": outlier_mode,
        "outliers_mitigados_total": int(outliers_total),
    }
    return regua, meta

# =========================
# UI
# =========================
st.title("Régua de Preço")
st.caption("Limpeza conservadora (remoção de linhas) + régua calculada em cima do tratado.")

with st.sidebar:
    st.subheader("Arquivo")
    uploaded = st.file_uploader("CSV do Google Shopping", type=["csv"])

    st.subheader("Configuração")
    sep_in_opt = st.selectbox("Separador do arquivo", ["Auto", ";", ","], index=0)
    sep_out = st.selectbox("Separador de exportação", [";", ","], index=0)
    aplicar_limpeza = st.checkbox("Aplicar limpeza antes da régua", value=True)

    st.subheader("Critério de Outlier")
    outlier_mode = st.selectbox(
        "Tratamento de outliers (apenas no cálculo da régua)",
        ["Nenhum", "Winsor P1/P99", "IQR 1.5x", "IQR 3.0x", "MAD z=3.5"],
        index=1,
        help="Não altera a base tratada. Afeta somente os cálculos da régua."
    )

    st.subheader("Exibição")
    modo_apresentacao = st.toggle("Modo apresentação", value=False)

if not uploaded:
    st.info("Anexe um CSV no menu lateral para iniciar.")
    st.stop()

raw = uploaded.getvalue()

if sep_in_opt == "Auto":
    sample = raw[:5000].decode("utf-8", errors="ignore")
    sep_in = ";" if sample.count(";") > sample.count(",") else ","
else:
    sep_in = sep_in_opt

df = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False, sep=sep_in)

if aplicar_limpeza:
    df_tratado, df_removidos, meta_clean = clean_df(df)
else:
    df_tratado, df_removidos = df.copy(), df.iloc[0:0].copy()
    meta_clean = {
        "linhas_original": int(len(df)),
        "linhas_tratado": int(len(df)),
        "linhas_removidos": 0,
        "top_motivos": {},
        "observacao": "Limpeza desativada."
    }

BASE_TRATADA_COLS = ["capturado_em", "palavra_chave", "produto", "marca", "seller", "preco", "rating", "reviews", "link"]
BASE_TRATADA_RENAME = {
    "capturado_em": "Data",
    "palavra_chave": "Categorias",
    "produto": "Produto",
    "marca": "Marca",
    "seller": "Loja",
    "preco": "Preço",
    "rating": "Rating",
    "reviews": "Qtd. Reviews",
    "link": "Link Google",
}
cols_existentes = [c for c in BASE_TRATADA_COLS if c in df_tratado.columns]
df_tratado_view = df_tratado[cols_existentes].rename(columns=BASE_TRATADA_RENAME)

regua, meta_regua = build_regua(df_tratado, outlier_mode=outlier_mode)
regua_exec = regua[["palavra_chave", "faixa_economica", "faixa_media", "faixa_premium"]].copy()

# Métricas rápidas (topo)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Linhas (Original)", f"{meta_clean['linhas_original']}")
k2.metric("Linhas (Tratado)", f"{meta_clean['linhas_tratado']}")
k3.metric("Removidos", f"{meta_clean['linhas_removidos']}")
k4.metric("Preços válidos", f"{meta_regua['precos_validos']}")

st.divider()

# =========================
# MODO APRESENTAÇÃO
# =========================
if modo_apresentacao:
    top_left, top_right = st.columns([1.05, 1.35])

    with top_left:
        st.subheader("Resumo")
        if "palavra_chave" in df.columns:
            vol = df["palavra_chave"].replace("", "(vazio)").value_counts().reset_index()
            vol.columns = ["palavra_chave", "linhas"]
            st.caption("Volume por palavra-chave")
            st.dataframe(vol, use_container_width=True, height=320)

        if meta_clean.get("observacao"):
            st.caption(meta_clean["observacao"])

        if meta_clean["top_motivos"]:
            motivos_df = pd.DataFrame(
                [{"motivo": k, "qtd": v} for k, v in meta_clean["top_motivos"].items()]
            ).sort_values("qtd", ascending=False)

            st.caption("Motivos de remoção (top)")
            st.dataframe(motivos_df, use_container_width=True, height=220)

    with top_right:
        st.subheader("Régua de Preço")
        st.caption("Visão enxuta: somente faixas Econômica / Média / Premium.")
        st.caption(f"Método de outliers (régua): {meta_regua.get('metodo_outlier', outlier_mode)} | Outliers mitigados (total): {meta_regua.get('outliers_mitigados_total', 0)}")
        st.dataframe(regua_exec, use_container_width=True, height=560)

    st.divider()
    st.subheader("Downloads (CSV)")

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        st.download_button(
            "Baixar Tratado",
            data=df_to_csv_bytes(df_tratado, sep=sep_out),
            file_name=f"{uploaded.name.replace('.csv','')}_tratado.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "Baixar Removidos",
            data=df_to_csv_bytes(df_removidos, sep=sep_out),
            file_name=f"{uploaded.name.replace('.csv','')}_removidos.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d3:
        st.download_button(
            "Baixar Régua",
            data=df_to_csv_bytes(regua, sep=sep_out),
            file_name=f"{uploaded.name.replace('.csv','')}_regua_preco.csv",
            mime="text/csv",
            use_container_width=True
        )

    with st.expander("Auditoria (opcional)", expanded=False):
        st.caption("Preview do CSV original")
        st.dataframe(df.head(30), use_container_width=True)

# =========================
# MODO NORMAL
# =========================
else:
    tab_visao, tab_tratado, tab_removidos, tab_regua = st.tabs(
        ["Visão Geral", "Base Tratada", "Removidos", "Régua de Preço"]
    )

    with tab_visao:
        st.subheader("Visão Geral")

        total_original = meta_clean["linhas_original"]
        total_tratado = meta_clean["linhas_tratado"]
        total_remov = meta_clean["linhas_removidos"]
        pct_remov = (total_remov / total_original * 100) if total_original else 0

        has_kw = "palavra_chave" in df_tratado.columns
        has_seller = "seller" in df_tratado.columns
        has_brand = "marca" in df_tratado.columns
        has_price = "preco" in df_tratado.columns

        if has_price:
            pnum = df_tratado["preco"].map(parse_price_to_float)
            p_valid = pnum.dropna()
            med = float(p_valid.median()) if len(p_valid) else np.nan
            p25 = float(p_valid.quantile(0.25)) if len(p_valid) else np.nan
            p75 = float(p_valid.quantile(0.75)) if len(p_valid) else np.nan
        else:
            p_valid = pd.Series([], dtype=float)
            med = p25 = p75 = np.nan

        # ==========================================================
        # NOVO: PAINEL EXECUTIVO (visão de decisão)
        # ==========================================================
        st.markdown("**Painel Executivo (visão de decisão)**")

        treated = int(total_tratado) if total_tratado else 0

        preco_validos = int(meta_regua.get("precos_validos", 0))
        preco_invalidos = int(meta_regua.get("precos_invalidos", 0))
        pct_preco_valido = (preco_validos / treated * 100) if treated else 0.0

        def pct_preenchido(col: str) -> float:
            if col not in df_tratado.columns or treated == 0:
                return np.nan
            s = df_tratado[col].astype(str).str.strip()
            return float((s != "").mean() * 100)

        pct_link = pct_preenchido("link")
        pct_marca = pct_preenchido("marca")
        pct_loja = pct_preenchido("seller")

        periodo_txt = "-"
        if "capturado_em" in df_tratado.columns:
            dt = pd.to_datetime(df_tratado["capturado_em"], errors="coerce", dayfirst=True)
            dt = dt.dropna()
            if len(dt):
                dmin, dmax = dt.min(), dt.max()
                if dmin.date() == dmax.date():
                    periodo_txt = dmin.strftime("%d/%m/%Y")
                else:
                    periodo_txt = f"{dmin.strftime('%d/%m')}–{dmax.strftime('%d/%m/%Y')}"

        top3_share = np.nan
        top1_share = np.nan
        n80 = np.nan
        lojas_unicas = np.nan

        if has_seller and treated > 0:
            vc = df_tratado["seller"].replace("", "(vazio)").value_counts()
            base = int(vc.sum()) if int(vc.sum()) > 0 else 1

            lojas_unicas = int(
                df_tratado["seller"].astype(str).str.strip().replace("", np.nan).dropna().nunique()
            )

            top1_share = float((int(vc.head(1).sum()) / base) * 100) if len(vc) else 0.0
            top3_share = float((int(vc.head(3).sum()) / base) * 100) if len(vc) else 0.0

            cum = (vc.cumsum() / base)
            n80 = int((cum <= 0.80).sum() + 1) if len(cum) else np.nan

        iqr = np.nan
        outlier_pct = np.nan
        outlier_qtd = 0

        if has_price and len(p_valid) >= 10:
            iqr = float(p75 - p25) if (not np.isnan(p75) and not np.isnan(p25)) else np.nan
            if not np.isnan(iqr):
                low = float(p25 - 1.5 * iqr)
                high = float(p75 + 1.5 * iqr)
                out = (p_valid < low) | (p_valid > high)
                outlier_qtd = int(out.sum())
                outlier_pct = float(outlier_qtd / len(p_valid) * 100) if len(p_valid) else np.nan

        st.caption("Qualidade da Base")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Base comparável", f"{treated}", f"{pct_remov:.1f}% removida" if total_original else None)
        q2.metric("Cobertura de preço", f"{pct_preco_valido:.0f}%", f"{preco_invalidos} sem preço" if preco_invalidos else "OK")
        q3.metric("Período capturado", periodo_txt)

        if not np.isnan(pct_link):
            q4.metric("Links preenchidos", f"{pct_link:.0f}%", "OK" if pct_link >= 85 else "Atenção")
        elif not np.isnan(pct_marca):
            q4.metric("Marcas preenchidas", f"{pct_marca:.0f}%", "OK" if pct_marca >= 85 else "Atenção")
        else:
            q4.metric("Atributos chave", "-", "Verificar colunas")

        st.caption("Concentração de Canal")
        c1, c2, c3, c4 = st.columns(4)
        if not np.isnan(top1_share):
            c1.metric("Top 1 loja (share)", f"{top1_share:.1f}%")
        else:
            c1.metric("Top 1 loja (share)", "-", "Sem seller")

        if not np.isnan(top3_share):
            c2.metric("Top 3 lojas (share)", f"{top3_share:.1f}%", "Risco" if top3_share >= 50 else "Saudável")
        else:
            c2.metric("Top 3 lojas (share)", "-", "Sem seller")

        if not np.isnan(n80):
            c3.metric("Lojas p/ 80% do volume", f"{n80}")
        else:
            c3.metric("Lojas p/ 80% do volume", "-", "Sem seller")

        if not np.isnan(lojas_unicas):
            c4.metric("Lojas únicas", f"{lojas_unicas}")
        else:
            c4.metric("Lojas únicas", "-", "Sem seller")

        st.caption("Sinais de Preço")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Ticket mediano", brl(med))
        p2.metric("Dispersão (IQR)", brl(iqr) if not np.isnan(iqr) else "-", "P75–P25")
        if not np.isnan(outlier_pct):
            p3.metric("Outliers de preço", f"{outlier_pct:.1f}%", f"{outlier_qtd} itens")
        else:
            p3.metric("Outliers de preço", "-", "Amostra pequena")

        alertas = []
        if treated and pct_preco_valido < 90:
            alertas.append("A cobertura de preço numérico está abaixo do ideal. A régua pode ficar instável.")
        if not np.isnan(pct_link) and pct_link < 80:
            alertas.append("Baixa cobertura de link. Dificulta auditoria e remoção de duplicidade por URL.")
        if not np.isnan(top3_share) and top3_share >= 50:
            alertas.append("Alta concentração em poucas lojas. A régua pode refletir mais o marketplace dominante do que o mercado.")
        if not np.isnan(outlier_pct) and outlier_pct >= 5:
            alertas.append("Cauda de outliers relevante. Provável mistura de kits/multipacks e itens fora do padrão.")

        if alertas:
            st.warning("**Alertas executivos:** " + " ".join([f"• {a}" for a in alertas]))
        else:
            st.success("Base com qualidade e concentração dentro do esperado para leitura executiva.")

        st.divider()

        # =========================
        # Tabelas rápidas (apoio)
        # =========================
        col1, col2, col3 = st.columns([1, 1.2, 1])

        with col1:
            st.markdown("**Top Palavras-chave (Tratado)**")
            if has_kw:
                top_kw = (
                    df_tratado["palavra_chave"]
                    .replace("", "(vazio)")
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                top_kw.columns = ["Palavra-chave", "Linhas"]
                st.dataframe(top_kw, use_container_width=True, height=320)
            else:
                st.caption("Coluna 'palavra_chave' não encontrada.")

        with col2:
            st.markdown("**Top Lojas (Pareto — Tratado)**")
            if has_seller:
                vc = df_tratado["seller"].replace("", "(vazio)").value_counts()
                base = int(vc.sum()) if int(vc.sum()) > 0 else 1

                pareto = vc.head(12).reset_index()
                pareto.columns = ["Loja", "Linhas"]
                pareto["Linhas"] = pd.to_numeric(pareto["Linhas"], errors="coerce").fillna(0).astype(int)

                pareto["% Base"] = (pareto["Linhas"] / base * 100).round(1)
                pareto["% Acumulado"] = pareto["% Base"].cumsum().round(1)

                st.dataframe(pareto, use_container_width=True, height=320)
            else:
                st.caption("Coluna 'seller' não encontrada.")

        with col3:
            st.markdown("**Top Marcas (Tratado)**")
            if has_brand:
                top_brand = (
                    df_tratado["marca"]
                    .replace("", "(vazio)")
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                top_brand.columns = ["Marca", "Linhas"]
                st.dataframe(top_brand, use_container_width=True, height=320)
            else:
                st.caption("Coluna 'marca' não encontrada.")

        st.divider()

        st.subheader("Downloads (CSV)")
        d1, d2, d3 = st.columns([1, 1, 1])

        with d1:
            st.download_button(
                "Baixar Tratado",
                data=df_to_csv_bytes(df_tratado, sep=sep_out),
                file_name=f"{uploaded.name.replace('.csv','')}_tratado.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            st.download_button(
                "Baixar Removidos",
                data=df_to_csv_bytes(df_removidos, sep=sep_out),
                file_name=f"{uploaded.name.replace('.csv','')}_removidos.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d3:
            st.download_button(
                "Baixar Régua",
                data=df_to_csv_bytes(regua, sep=sep_out),
                file_name=f"{uploaded.name.replace('.csv','')}_regua_preco.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab_tratado:
        st.subheader("Base Tratada")
        st.caption("Visão executiva: colunas-chave para leitura em reunião.")
        st.dataframe(df_tratado_view, use_container_width=True)

    with tab_removidos:
        st.subheader("Removidos")
        if len(df_removidos) == 0:
            st.success("Nenhuma linha removida nesta execução.")
        else:
            st.dataframe(df_removidos, use_container_width=True)

    with tab_regua:
        st.subheader("Régua de Preço")
        st.caption("Visão enxuta: somente faixas Econômica / Média / Premium.")
        st.caption(f"Método de outliers (régua): {meta_regua.get('metodo_outlier', outlier_mode)} | Outliers mitigados (total): {meta_regua.get('outliers_mitigados_total', 0)}")
        st.dataframe(regua_exec, use_container_width=True)
