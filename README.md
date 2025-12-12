# Pesquisa de Mercado — Google Shopping (Python)

Projeto em Python para **coletar e processar resultados do Google Shopping** por palavra-chave, salvando saídas em **CSV** e oferecendo um **dashboard** para exploração.

---

## Estrutura do projeto

```text
pesquisa_mercado/
├─ outputs/                    # arquivos gerados (CSVs)
├─ referenciais/               # referências (ex.: marcas conhecidas)
│  ├─ __init__.py
│  └─ marcas_conhecidas.py
├─ .env                        # variáveis de ambiente (não versionar)
├─ config.py
├─ dashboard.py
├─ fuzzy_matching.py
├─ google_shopping_client.py
├─ main.py
└─ processador_resultados.py
```

---

## Pré-requisitos

- Python 3.10+ (recomendado)
- Ambiente virtual (recomendado)

---

## Instalação

### 1) Criar e ativar o ambiente virtual

**Windows**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Instalar dependências

> Use o arquivo de dependências que existir no seu repositório (ex.: `requirements.txt` ou `pyproject.toml`).

Exemplo (`requirements.txt`):
```bash
pip install -r requirements.txt
```

---

## Configuração do `.env`

- **Não commitar** o arquivo `.env` (ele pode conter chaves/segredos).
- Mantenha um `.env.example` com **os nomes das variáveis** (sem valores reais).

Exemplo de `.env.example` (ajuste para as suas variáveis reais):
```env
OUTPUT_DIR=outputs
# EXEMPLOS (se aplicável ao seu fluxo)
# GOOGLE_API_KEY=
# DEFAULT_KEYWORD=
```

---

## Como executar

### Rodar o fluxo principal

```bash
python main.py
```

As saídas (CSVs) serão gravadas em `outputs/`.

---

## Dashboard (Streamlit)

Para iniciar o dashboard:

```bash
streamlit run dashboard.py
```

---

## Saídas geradas

Os resultados ficam em `outputs/` com nomes no padrão:

- `resultado_google_shopping_<palavra_chave>.csv`

---

## Boas práticas (Git)

- Versione o código (`*.py`) e arquivos de configuração não sensíveis.
- Ignore: `.env`, `.venv/`, `__pycachecache__/`, `__pycache__/` e saídas regeneráveis em `outputs/`.

