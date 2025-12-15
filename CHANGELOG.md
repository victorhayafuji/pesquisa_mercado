# Changelog

## v1.1 (2025-12)
- Mantido o robô de coleta do Google Shopping via SearchAPI (CSV como saída principal).
- Mantido o algoritmo de score na main.py para enriquecer o ranking no próprio CSV.
- Adicionado dashboard em Streamlit (dashboard.py) para consumo interno do CSV:
  - Leitura por seleção de arquivo em outputs/ ou upload manual.
  - Resumo executivo em 1 frase (com preço médio) e seções analíticas.
