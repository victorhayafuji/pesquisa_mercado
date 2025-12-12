"""
Dashboard de Inteligência de Mercado (Google Shopping) a partir de CSV já existente.

Fluxo:
1) Usuário seleciona um CSV salvo em OUTPUT_DIR ou faz upload de um CSV.
2) Lemos o arquivo.
3) Construímos o dashboard em uma única página, seguindo o framework:
   - Introdução e objetivo
   - Metodologia e base de dados
   - Estrutura de preços
   - Estrutura de sellers
   - Estrutura de marcas
   - Reputação (rating e reviews)
   - Limitações
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import altair as alt
import pandas as pd
import streamlit as st

from config import OUTPUT_DIR  # mesma pasta onde os CSVs são salvos


# =========================
# Configuração global e tema
# =========================

st.set_page_config(
    page_title="Inteligência de Mercado - Google Shopping",
    layout="wide",
)


def inject_css() -> None:
    """Tema claro + componentes simples de UI (cards, containers)."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f5f7;
            --panel: #ffffff;
            --panel-soft: #f9fafb;
            --border-subtle: #d0d4dd;
            --accent: #2563eb;
            --accent-soft: rgba(37,99,235,0.06);
            --text-main: #111827;
            --text-muted: #6b7280;
        }

        .stApp {
            background-color: var(--bg);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        h1, h2, h3, h4 {
            color: var(--text-main);
        }

        .metric-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .metric-card {
            background: var(--panel);
            border-radius: 0.75rem;
            border: 1px solid var(--border-subtle);
            padding: 0.8rem 1rem;
            min-width: 180px;
            flex: 1 1 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 0.15rem;
            color: var(--text-main);
        }

        .metric-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }

        .panel-box {
            background: var(--panel-soft);
            border-radius: 0.75rem;
            border: 1px solid var(--border-subtle);
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def styled_chart(chart: alt.Chart) -> alt.Chart:
    """Aplica estilo visual consistente (fundo claro, texto escuro)."""
    return (
        chart.configure_view(
            strokeWidth=0,
            fill="#ffffff",
        )
        .configure_axis(
            labelColor="#111827",
            titleColor="#111827",
            labelFontSize=12,
            titleFontSize=13,
            labelFontWeight="bold",
            titleFontWeight="bold",
            grid=True,
            gridColor="#e5e7eb",
        )
        .configure_title(
            color="#111827",
            fontSize=16,
            fontWeight="bold",
            anchor="start",
        )
        .configure_legend(
            labelColor="#111827",
            titleColor="#111827",
            labelFontSize=11,
            titleFontSize=12,
            labelFontWeight="bold",
            titleFontWeight="bold",
        )
    )


def metric_card(label: str, value: str, subtitle: Optional[str] = None) -> None:
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def preparar_bases(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Converte colunas numéricas e cria base sem outliers, quando possível."""
    df_num = df.copy()

    num_cols = [
        "preco",
        "preco_original",
        "desconto_percentual",
        "indice_preco",
        "score_preco",
        "score_promo",
        "score_atratividade",
        "score_visibilidade",
        "score_qualidade",
        "score_relevancia",
        "rating",
        "reviews",
        "posicao",
    ]
    for col in num_cols:
        if col in df_num.columns:
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    if "capturado_em" in df_num.columns:
        try:
            df_num["capturado_em_dt"] = pd.to_datetime(df_num["capturado_em"], errors="coerce")
        except Exception:
            df_num["capturado_em_dt"] = pd.NaT
    else:
        df_num["capturado_em_dt"] = pd.NaT

    if "preco" in df_num.columns and "preco_outlier" in df_num.columns:
        base_core = df_num[(df_num["preco"].notna()) & (~df_num["preco_outlier"])]
    elif "preco" in df_num.columns:
        base_core = df_num[df_num["preco"].notna()]
    else:
        base_core = df_num

    return df_num, base_core


def resumo_preco(series: pd.Series) -> Optional[Dict[str, float]]:
    """Retorna dicionário com estatísticas básicas de preço."""
    s = series.dropna()
    if s.empty:
        return None
    return {
        "n": float(len(s)),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "mediana": float(s.median()),
        "media": float(s.mean()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def format_money(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==============================
# Painel gráfico / Storytelling
# ==============================

def render_dashboard(df: pd.DataFrame, palavra_chave: Optional[str], caminho_arquivo: Optional[str]) -> None:
    df_num, base_core = preparar_bases(df)

    # ---------- 1. Introdução e Objetivo ----------
    st.markdown("## 1. Introdução e objetivo")

    col_i1, col_i2 = st.columns([3, 2])
    with col_i1:
        st.markdown(
            f"""
            **Categoria analisada:** `{palavra_chave or "não informado"}`  
            
            A unidade de análise é o anúncio individual no Google Shopping  
            (um produto em um seller específico).  
            O objetivo é entender faixas de preço, presença de sellers e marcas  
            e reputação (rating e reviews) para apoiar decisões de Produto, Comercial,  
            Marketing e Pricing.
            """
        )
    with col_i2:
        if "capturado_em_dt" in df_num.columns and df_num["capturado_em_dt"].notna().any():
            dt_min = df_num["capturado_em_dt"].min()
            dt_max = df_num["capturado_em_dt"].max()
            st.markdown(
                f"""
                **Período de captura**  
                - Data mínima: `{dt_min.strftime("%d/%m/%Y %H:%M")}`  
                - Data máxima: `{dt_max.strftime("%d/%m/%Y %H:%M")}`  
                """
            )
        else:
            st.markdown("**Período de captura:** não disponível na base.")

        if caminho_arquivo:
            st.caption(f"Arquivo analisado: `{caminho_arquivo}`")

    st.markdown("---")

    # ---------- 2. Metodologia e base de dados ----------
    st.markdown("## 2. Metodologia e base de dados")

    total_linhas = len(df_num)
    n_sellers = df_num["seller"].nunique() if "seller" in df_num.columns else 0
    n_marcas = df_num["marca"].nunique() if "marca" in df_num.columns else 0

    preco_mediana_core = None
    pct_promo = None
    if "preco" in base_core.columns:
        preco_mediana_core = base_core["preco"].dropna().median()
    if "desconto_percentual" in df_num.columns:
        base_desc = df_num["desconto_percentual"].dropna()
        if not base_desc.empty:
            pct_promo = (base_desc.gt(0).mean()) * 100.0

    # Linha de cards de resumo
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    metric_card("Total Anúncios", f"{total_linhas:,}".replace(",", "."))
    metric_card("Total Sellers" , f"{n_sellers:,}".replace(",", "."))
    metric_card("Total Marcas", f"{n_marcas:,}".replace(",", "."))
    if preco_mediana_core is not None and not math.isnan(preco_mediana_core):
        metric_card("Mediana Preço (sem outliers)", format_money(preco_mediana_core))
    if pct_promo is not None and not math.isnan(pct_promo):
        metric_card("% de anúncios em promoção", f"{pct_promo:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Qualidade dos dados
    st.markdown("### 2.1 Qualidade dos dados")

    campos_qualidade: List[Dict[str, Any]] = []
    for col in ["marca", "rating", "reviews", "preco", "preco_outlier"]:
        if col in df_num.columns:
            n_missing = int(df_num[col].isna().sum())
            campos_qualidade.append(
                {
                    "Campo": col,
                    "Valores nulos": n_missing,
                    "% nulos": round(100 * n_missing / len(df_num), 1),
                }
            )
    if campos_qualidade:
        qual_df = pd.DataFrame(campos_qualidade)
        st.dataframe(qual_df, hide_index=True)
    else:
        st.write("Não foi possível calcular qualidade de dados: campos esperados não encontrados.")

    st.markdown("---")

    # ---------- 3. Análise descritiva da categoria ----------
    st.markdown("## 3. Análise descritiva da categoria")

    # ----- 3.1 Estrutura de preços -----
    st.markdown("### 3.1 Estrutura de preços")

    if "preco" in df_num.columns:
        df_preco_all = df_num[df_num["preco"].notna()]
        df_preco_core = base_core[base_core["preco"].notna()]

        resumo_all = resumo_preco(df_preco_all["preco"])
        resumo_core = resumo_preco(df_preco_core["preco"])

        col_pa, col_pb = st.columns(2)

        with col_pa:
            st.markdown("Base completa (com outliers)")
            if resumo_all:
                st.markdown(
                    f"""
    - N = {int(resumo_all["n"])} anúncios  
    - Mínimo: {format_money(resumo_all["min"])}  
    - Q1: {format_money(resumo_all["q1"])}  
    - Mediana: {format_money(resumo_all["mediana"])}  
    - Média: {format_money(resumo_all["media"])}  
    - Q3: {format_money(resumo_all["q3"])}  
    - Máximo: {format_money(resumo_all["max"])}  
    """
                )
            else:
                st.write("Sem dados válidos de preço.")

        with col_pb:
            st.markdown("Base sem outliers de preço")
            if resumo_core:
                st.markdown(
                    f"""
    - N = {int(resumo_core["n"])} anúncios  
    - Mínimo: {format_money(resumo_core["min"])}  
    - Q1: {format_money(resumo_core["q1"])}  
    - Mediana: {format_money(resumo_core["mediana"])}  
    - Média: {format_money(resumo_core["media"])}  
    - Q3: {format_money(resumo_core["q3"])}  
    - Máximo: {format_money(resumo_core["max"])}  
    """
                )
            else:
                st.write("Sem dados suficientes sem outliers.")

        # === PARETO DE FAIXAS DE PREÇO ===
        if not df_preco_core.empty:
            s = df_preco_core["preco"].dropna()
            if not s.empty:
                df_bins = s.to_frame(name="preco")
                n_bins = 8  # número de faixas de preço
                df_bins["faixa"] = pd.cut(df_bins["preco"], bins=n_bins)

                agg = (
                    df_bins.groupby("faixa")
                    .size()
                    .reset_index(name="qtd")
                )
                agg = agg[agg["faixa"].notna()]

                # Label legível da faixa
                def label_faixa(intervalo):
                    left = intervalo.left
                    right = intervalo.right
                    return (
                        f"R$ {left:,.0f} – R$ {right:,.0f}"
                        .replace(",", ".")
                    )

                agg["faixa_label"] = agg["faixa"].apply(label_faixa)

                # Ordena por quantidade (Pareto)
                agg = agg.sort_values("qtd", ascending=False)
                agg["pct"] = agg["qtd"] / agg["qtd"].sum() * 100
                agg["pct_acum"] = agg["pct"].cumsum()

                base_chart = alt.Chart(agg).encode(
                    x=alt.X(
                        "faixa_label:N",
                        sort="-y",
                        title="Faixa de preço (ordenada por frequência)",
                        axis=alt.Axis(labelAngle=-45),
                    )
                )

                # Barras de quantidade – azul
                bars = base_chart.mark_bar(color="#2563eb").encode(
                    y=alt.Y("qtd:Q", title="Quantidade de anúncios"),
                    tooltip=[
                        alt.Tooltip("faixa_label:N", title="Faixa de preço"),
                        alt.Tooltip("qtd:Q", title="Qtde anúncios"),
                        alt.Tooltip("pct_acum:Q", title="% acumulado", format=".1f"),
                    ],
                )

                # Valores em cima das barras
                text_bars = base_chart.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-4,
                    fontWeight="bold",
                    fontSize=11,
                    color="#111827",
                ).encode(
                    y="qtd:Q",
                    text="qtd:Q",
                )

                # Linha de % acumulado
                linha = base_chart.mark_line(
                    strokeWidth=2,
                    point=True,
                    color="#000000",
                ).encode(
                    y=alt.Y(
                        "pct_acum:Q",
                        axis=None,  # <<< remove totalmente o eixo direito
                        scale=alt.Scale(domain=[0, 100]),
                    )
                )

                # Valores em cima dos pontos da linha de Pareto (% acumulado)
                text_linha = base_chart.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-6,
                    fontSize=10,
                    fontWeight="bold",
                    color="#000000",
                ).encode(
                    y=alt.Y("pct_acum:Q", axis=None),  # <<< garante que esse layer também não cria eixo
                    text=alt.Text("pct_acum:Q", format=".0f"),
                )

                chart_pareto = alt.layer(
                    bars,
                    text_bars,
                    linha,
                    text_linha,
                ).resolve_scale(y="independent")

                # Área maior do gráfico (altura)
                chart_pareto = chart_pareto.properties(
                    height=380,
                    title="Pareto de faixas de preço (base sem outliers)",
                )

                st.altair_chart(styled_chart(chart_pareto), use_container_width=True)

        if "preco_outlier" in df_num.columns:
            n_out = int(df_num["preco_outlier"].fillna(False).sum())
            if n_out > 0:
                st.caption(f"{n_out} anúncios marcados como outliers de preço (preco_outlier = True).")

    # ----- 3.2 Estrutura de sellers -----
    st.markdown("### 3.2 Estrutura de sellers (lojistas)")

    if "seller" in base_core.columns and "preco" in base_core.columns:
        df_s = base_core.dropna(subset=["seller", "preco"]).copy()

        agg_sellers = (
            df_s.groupby("seller")
            .agg(
                qtd_anuncios=("seller", "size"),
                preco_medio=("preco", "mean"),
                preco_mediano=("preco", "median"),
                preco_min=("preco", "min"),
                preco_max=("preco", "max"),
                rating_medio=("rating", "mean") if "rating" in df_s.columns else ("preco", "size"),
                reviews_totais=("reviews", "sum") if "reviews" in df_s.columns else ("preco", "size"),
            )
            .reset_index()
        )

        if "rating" not in df_s.columns:
            agg_sellers = agg_sellers.rename(columns={"rating_medio": "rating_medio (não disponível)"})
        if "reviews" not in df_s.columns:
            agg_sellers = agg_sellers.rename(columns={"reviews_totais": "reviews_totais (não disponível)"})

        for col in ["preco_medio", "preco_mediano", "preco_min", "preco_max", "rating_medio"]:
            if col in agg_sellers.columns:
                agg_sellers[col] = agg_sellers[col].astype(float).round(2)
        if "reviews_totais" in agg_sellers.columns:
            agg_sellers["reviews_totais"] = pd.to_numeric(agg_sellers["reviews_totais"], errors="coerce").fillna(0).astype(int)

        top_sellers = agg_sellers.sort_values("qtd_anuncios", ascending=False).head(10)

        cs1, cs2 = st.columns([2, 3])

        with cs1:
            st.markdown("Top 10 sellers por quantidade de anúncios")
            base_chart = alt.Chart(top_sellers).encode(
                y=alt.Y("seller:N", sort="-x", title="Seller"),
                x=alt.X("qtd_anuncios:Q", title="Quantidade de anúncios"),
            )
            bars = base_chart.mark_bar()
            labels = base_chart.mark_text(
                align="left",
                baseline="middle",
                dx=3,
                color="#111827",
            ).encode(text="qtd_anuncios:Q")
            chart_sellers = (bars + labels).properties(height=280)
            st.altair_chart(styled_chart(chart_sellers), use_container_width=True)

        with cs2:
            st.markdown("Resumo de preço e reputação por seller (Top 10)")
            cols_show = [c for c in [
                "seller",
                "qtd_anuncios",
                "preco_medio",
                "preco_mediano",
                "preco_min",
                "preco_max",
                "rating_medio",
                "reviews_totais",
            ] if c in top_sellers.columns]
            st.dataframe(top_sellers[cols_show].reset_index(drop=True))
    else:
        st.info("Não há colunas suficientes para analisar sellers (necessário seller e preco).")

    st.markdown("---")

    # ----- 3.3 Estrutura de marcas -----
    st.markdown("### 3.3 Estrutura de marcas")

    if "marca" in base_core.columns and "preco" in base_core.columns:
        df_m = base_core.dropna(subset=["preco"]).copy()

        agg_marcas = (
            df_m.groupby("marca", dropna=False)
            .agg(
                qtd_anuncios=("marca", "size"),
                preco_medio=("preco", "mean"),
                preco_mediano=("preco", "median"),
                preco_min=("preco", "min"),
                preco_max=("preco", "max"),
                rating_medio=("rating", "mean") if "rating" in df_m.columns else ("preco", "size"),
                reviews_totais=("reviews", "sum") if "reviews" in df_m.columns else ("preco", "size"),
            )
            .reset_index()
        )

        agg_marcas["marca"] = agg_marcas["marca"].fillna("SEM MARCA")

        if "rating" not in df_m.columns:
            agg_marcas = agg_marcas.rename(columns={"rating_medio": "rating_medio (não disponível)"})
        if "reviews" not in df_m.columns:
            agg_marcas = agg_marcas.rename(columns={"reviews_totais": "reviews_totais (não disponível)"})

        for col in ["preco_medio", "preco_mediano", "preco_min", "preco_max", "rating_medio"]:
            if col in agg_marcas.columns:
                agg_marcas[col] = agg_marcas[col].astype(float).round(2)
        if "reviews_totais" in agg_marcas.columns:
            agg_marcas["reviews_totais"] = pd.to_numeric(agg_marcas["reviews_totais"], errors="coerce").fillna(0).astype(int)

        top_marcas = agg_marcas.sort_values("qtd_anuncios", ascending=False).head(10)

        cm1, cm2 = st.columns([2, 3])

        with cm1:
            st.markdown("Top 10 marcas por quantidade de anúncios")
            base_chart_m = alt.Chart(top_marcas).encode(
                y=alt.Y("marca:N", sort="-x", title="Marca"),
                x=alt.X("qtd_anuncios:Q", title="Quantidade de anúncios"),
            )
            bars_m = base_chart_m.mark_bar()
            labels_m = base_chart_m.mark_text(
                align="left",
                baseline="middle",
                dx=3,
                color="#111827",
            ).encode(text="qtd_anuncios:Q")
            chart_marcas = (bars_m + labels_m).properties(height=280)
            st.altair_chart(styled_chart(chart_marcas), use_container_width=True)

        with cm2:
            st.markdown("Resumo de preço e reputação por marca (Top 10)")
            cols_show_m = [c for c in [
                "marca",
                "qtd_anuncios",
                "preco_medio",
                "preco_mediano",
                "preco_min",
                "preco_max",
                "rating_medio",
                "reviews_totais",
            ] if c in top_marcas.columns]
            st.dataframe(top_marcas[cols_show_m].reset_index(drop=True))
    else:
        st.info("Não há colunas suficientes para analisar marcas (necessário marca e preco).")

    st.markdown("---")

    # ----- 3.4 Reputação e engajamento -----
    st.markdown("### 3.4 Reputação e engajamento (rating e reviews)")

    if "rating" in df_num.columns and "reviews" in df_num.columns:
        df_r = df_num.copy()
        df_r["rating"] = pd.to_numeric(df_r["rating"], errors="coerce")
        df_r["reviews"] = pd.to_numeric(df_r["reviews"], errors="coerce")

        base_valida = df_r.dropna(subset=["rating", "reviews"])

        if not base_valida.empty:
            rating_min = base_valida["rating"].min()
            rating_max = base_valida["rating"].max()
            rating_med = base_valida["rating"].mean()
            rating_mediana = base_valida["rating"].median()

            reviews_min = int(base_valida["reviews"].min())
            reviews_max = int(base_valida["reviews"].max())
            reviews_mediana = float(base_valida["reviews"].median())

            cr1, cr2 = st.columns(2)

            # --- Rating ---
            with cr1:
                st.markdown("**Distribuição de rating**")
                st.markdown(
                    f"""
            - Mínimo: {rating_min:.1f}  
            - Máximo: {rating_max:.1f}  
            - Média: {rating_med:.1f}  
            - Mediana: {rating_mediana:.1f}  
            """
                )

                # Rating categorizado e Pareto
                base_valida["rating_cat"] = base_valida["rating"].round(1).astype(str)
                dist_rating = (
                    base_valida.groupby("rating_cat")
                    .size()
                    .reset_index(name="qtd")
                )

                # Ordena por frequência (Pareto) e calcula % acumulado
                dist_rating = dist_rating.sort_values("qtd", ascending=False)
                dist_rating["pct"] = dist_rating["qtd"] / dist_rating["qtd"].sum() * 100
                dist_rating["pct_acum"] = dist_rating["pct"].cumsum()

                base_chart_r = alt.Chart(dist_rating).encode(
                    x=alt.X(
                        "rating_cat:N",
                        sort="-y",
                        title="Rating",
                    )
                )

                # Barras (quantidade de anúncios)
                bars_r = base_chart_r.mark_bar(color="#2563eb").encode(
                    y=alt.Y("qtd:Q", title="Quantidade de anúncios"),
                    tooltip=[
                        alt.Tooltip("rating_cat:N", title="Rating"),
                        alt.Tooltip("qtd:Q", title="Qtde anúncios"),
                        alt.Tooltip("pct_acum:Q", title="% acumulado", format=".1f"),
                    ],
                )

                # Valores sobre as barras
                text_bars_r = base_chart_r.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-4,
                    fontWeight="bold",
                    fontSize=11,
                    color="#111827",
                ).encode(
                    y="qtd:Q",
                    text="qtd:Q",
                )

                # Linha de % acumulado (Pareto) – roxa, sem eixo
                linha_r = base_chart_r.mark_line(
                    strokeWidth=2,
                    point=True,
                    color="#000000",
                ).encode(
                    y=alt.Y(
                        "pct_acum:Q",
                        axis=None,
                        scale=alt.Scale(domain=[0, 100]),
                    )
                )

                # Valores de % acumulado em cima da linha
                text_linha_r = base_chart_r.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-6,
                    fontSize=10,
                    fontWeight="bold",
                    color="#000000",
                ).encode(
                    y=alt.Y("pct_acum:Q", axis=None),
                    text=alt.Text("pct_acum:Q", format=".0f"),
                )

                chart_rating = alt.layer(
                    bars_r,
                    text_bars_r,
                    linha_r,
                    text_linha_r,
                ).resolve_scale(y="independent")

                chart_rating = chart_rating.properties(
                    height=320,
                    title="Pareto de rating (distribuição de anúncios)",
                )

                st.altair_chart(styled_chart(chart_rating), use_container_width=True)

                # --- Reviews ---
                with cr2:
                    st.markdown("**Distribuição de número de reviews**")
                    st.markdown(
                        f"""
            - Mínimo: {reviews_min}  
            - Máximo: {reviews_max}  
            - Mediana: {reviews_mediana:.1f}  
            """
                    )

                    # Agrupamento em faixas de reviews
                    def faixa_reviews(v: float) -> str:
                        if pd.isna(v):
                            return "Sem dados"
                        v = float(v)
                        if v <= 10:
                            return "1–10"
                        elif v <= 50:
                            return "11–50"
                        elif v <= 100:
                            return "51–100"
                        elif v <= 250:
                            return "101–250"
                        elif v <= 500:
                            return "251–500"
                        elif v <= 1000:
                            return "501–1000"
                        elif v <= 2000:
                            return "1001–2000"
                        else:
                            return ">2000"

                    base_valida["faixa_reviews"] = base_valida["reviews"].apply(faixa_reviews)

                    dist_reviews = (
                        base_valida.groupby("faixa_reviews")
                        .size()
                        .reset_index(name="qtd")
                    )

                    # Ordena por frequência (Pareto) e calcula % acumulado
                    dist_reviews = dist_reviews.sort_values("qtd", ascending=False)
                    dist_reviews["pct"] = dist_reviews["qtd"] / dist_reviews["qtd"].sum() * 100
                    dist_reviews["pct_acum"] = dist_reviews["pct"].cumsum()

                    base_chart_rev = alt.Chart(dist_reviews).encode(
                        x=alt.X(
                            "faixa_reviews:N",
                            sort="-y",
                            title="Faixa de número de reviews",
                            axis=alt.Axis(labelAngle=-45),
                        )
                    )

                    # Barras – quantidade de anúncios
                    bars_rev = base_chart_rev.mark_bar(color="#2563eb").encode(
                        y=alt.Y("qtd:Q", title="Quantidade de anúncios"),
                        tooltip=[
                            alt.Tooltip("faixa_reviews:N", title="Faixa de reviews"),
                            alt.Tooltip("qtd:Q", title="Qtde anúncios"),
                            alt.Tooltip("pct_acum:Q", title="% acumulado", format=".1f"),
                        ],
                    )

                    # Valores sobre as barras
                    text_bars_rev = base_chart_rev.mark_text(
                        align="center",
                        baseline="bottom",
                        dy=-4,
                        fontWeight="bold",
                        fontSize=11,
                        color="#111827",
                    ).encode(
                        y="qtd:Q",
                        text="qtd:Q",
                    )

                    # Linha de % acumulado – roxa, sem eixo
                    linha_rev = base_chart_rev.mark_line(
                        strokeWidth=2,
                        point=True,
                        color="#000000",
                    ).encode(
                        y=alt.Y(
                            "pct_acum:Q",
                            axis=None,
                            scale=alt.Scale(domain=[0, 100]),
                        )
                    )

                    # Valores da linha (% acumulado)
                    text_linha_rev = base_chart_rev.mark_text(
                        align="center",
                        baseline="bottom",
                        dy=-6,
                        fontSize=10,
                        fontWeight="bold",
                        color="#000000",
                    ).encode(
                        y=alt.Y("pct_acum:Q", axis=None),
                        text=alt.Text("pct_acum:Q", format=".0f"),
                    )

                    chart_reviews = alt.layer(
                        bars_rev,
                        text_bars_rev,
                        linha_rev,
                        text_linha_rev,
                    ).resolve_scale(y="independent")

                    chart_reviews = chart_reviews.properties(
                        height=320,
                        title="Pareto de faixas de reviews (distribuição de anúncios)",
                    )

                    st.altair_chart(styled_chart(chart_reviews), use_container_width=True)

            # Produtos âncora
            base_valida_sorted = base_valida.sort_values("reviews", ascending=False)
            produtos_ancora = base_valida_sorted[base_valida_sorted["rating"] >= 4.0].head(5)

            st.markdown("**Produtos âncora (mais reviews e boa nota)**")
            cols_anchor = [c for c in [
                "produto",
                "seller",
                "marca",
                "preco",
                "rating",
                "reviews",
            ] if c in produtos_ancora.columns]
            if not produtos_ancora.empty and cols_anchor:
                df_anchor = produtos_ancora[cols_anchor].copy()
                if "preco" in df_anchor.columns:
                    df_anchor["preco"] = df_anchor["preco"].apply(
                        lambda v: format_money(v) if pd.notna(v) else None
                    )
                st.dataframe(df_anchor.reset_index(drop=True))
            else:
                st.write("Não foram encontrados produtos com rating ≥ 4,0 e base relevante de reviews.")
        else:
            st.info("Não há base suficiente com rating e reviews preenchidos para análise de reputação.")
    else:
        st.info("Campos rating e reviews não encontrados; não é possível analisar reputação.")

    st.markdown("---")

    # ---------- 4. Limitações automáticas ----------
    st.markdown("## 4. Limitações automáticas da análise")

    limitacoes: List[str] = []

    if "capturado_em_dt" not in df_num.columns or df_num["capturado_em_dt"].isna().all():
        limitacoes.append("- Ausência de data/hora confiável de captura (capturado_em).")

    for campo in ["marca", "rating", "reviews"]:
        if campo in df_num.columns:
            pct_null = df_num[campo].isna().mean() * 100
            if pct_null > 30:
                limitacoes.append(
                    f"- Campo **{campo}** com {pct_null:.1f}% de valores nulos; interpretar estatísticas com cautela."
                )

    if "preco" not in df_num.columns:
        limitacoes.append("- Campo de preço (preco) ausente; análise de preços fica comprometida.")

    if not limitacoes:
        st.write("Não foram identificadas grandes limitações além de ser uma fotografia pontual do Google Shopping.")
    else:
        for item in limitacoes:
            st.write(item)

    st.caption(
        "Esta análise é uma fotografia pontual do Google Shopping, limitada à palavra-chave pesquisada "
        "e aos campos presentes no CSV."
    )


# ==============================
# Interface principal (seleção de CSV)
# ==============================

inject_css()
st.title("")
st.title("Pesquisa de Mercado – Google Shopping")

# Recupera estado anterior, se existir
df_state = st.session_state.get("df_resultado", None)
palavra_state = st.session_state.get("palavra_chave", None)
arquivo_state = st.session_state.get("arquivo_csv", None)

st.markdown("### Seleção da base de dados (CSV)")

col1, col2 = st.columns([2, 2])

# 0.1 – Seleção de arquivo salvo em OUTPUT_DIR
with col1:
    output_dir = Path(OUTPUT_DIR)
    csv_files = sorted(output_dir.glob("resultado_google_shopping_*.csv"))
    opcoes = ["-- selecione um arquivo da pasta OUTPUT_DIR --"] + [f.name for f in csv_files]
    escolha = st.selectbox("Arquivos disponíveis na pasta de saída", opcoes)

# 0.2 – Upload manual de CSV
with col2:
    uploaded = st.file_uploader("Ou carregue um CSV do seu computador", type=["csv"])

df = df_state
palavra_atual = palavra_state
arquivo_atual = arquivo_state

# Lógica de carregamento: prioridade para upload; se não houver, usa seleção da pasta
if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Erro ao ler CSV enviado: {e}")
    else:
        df = df_up
        arquivo_atual = uploaded.name
        # tenta inferir a palavra-chave do nome do arquivo, se seguir o padrão
        nome = Path(uploaded.name).stem
        if nome.startswith("resultado_google_shopping_"):
            palavra_atual = nome.replace("resultado_google_shopping_", "").replace("_", " ")
        else:
            palavra_atual = None
        st.session_state["df_resultado"] = df
        st.session_state["arquivo_csv"] = arquivo_atual
        st.session_state["palavra_chave"] = palavra_atual
        st.success(f"CSV carregado a partir do upload: {arquivo_atual}")

elif escolha != opcoes[0]:
    caminho = output_dir / escolha
    try:
        df_file = pd.read_csv(caminho, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Erro ao ler CSV da pasta OUTPUT_DIR: {e}")
    else:
        df = df_file
        arquivo_atual = str(caminho)
        nome = Path(caminho).stem
        if nome.startswith("resultado_google_shopping_"):
            palavra_atual = nome.replace("resultado_google_shopping_", "").replace("_", " ")
        else:
            palavra_atual = None
        st.session_state["df_resultado"] = df
        st.session_state["arquivo_csv"] = arquivo_atual
        st.session_state["palavra_chave"] = palavra_atual
        st.success(f"CSV carregado: {arquivo_atual}")

# Se há DataFrame carregado, mostra amostra e dashboard
if df is not None and not df.empty:
    st.subheader("Amostra dos dados (até 50 linhas)")
    st.dataframe(df.head(50))
    st.markdown("---")
    render_dashboard(df, palavra_atual, arquivo_atual)
else:
    st.info("Selecione ou carregue um CSV para visualizar o dashboard.")
