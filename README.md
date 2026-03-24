# Pesquisa de Inteligência B2B — Pricing (Google Shopping)

Aplicação robusta e autônoma desenvolvida em **Python + Streamlit** focado em **Inteligência de Pricing B2B**. O sistema extrai preços do mercado (Google Shopping), processa as regras de limpeza, formata uma régua estatística GBB (Good-Better-Best) e cruza os dados extraídos diretamente com o portfólio de produtos internos da empresa.

Tudo executado num ambiente puramente visual e fácil, além de possuir integração total para análises e Dashboards avançados no **Power BI** usando armazenamento local **SQLite**.

---

## 🏗️ Estrutura Atualizada do Projeto

O fluxo modular agora acontece majoritariamente ao redor de **`app_v2.py`** e usa **`database.py`** para rastreamento longitudinal temporal de todas pesquisas de mercado. 

```text
pesquisa_mercado/
├─ outputs/                    # Exportações sob demanda
├─ .env                        # Chaves de API (não versionar)
├─ historico_pricing.db        # Banco SQLite local (dados ODS PBI)
├─ Iniciar_App.bat             # Atalho nativo para inicialização com duplo clique!
├─ app_v2.py                   # 🔥 Nova Aplicação Principal (Streamlit)
├─ core_v2.py                  # Core Engine com extração e GBB rules
├─ database.py                 # Integração SQLite Auto-Migrável e Exports PBI
├─ config.py                   # Configurações de diretórios
├─ google_shopping_client.py   # Client REST para o Google
├─ regras_limpeza.json         # Dicionário de regras dinâmicas anti-sujeira
└─ processador_resultados.py   # Parsing das SERPs JSON da API
```

---

## 🚀 Como Executar

A complexidade do terminal tornou-se no passado! Agora o dashboard possui acionamento _One-Click_.

1. Vá até a sua Área de Trabalho e clique no **`Iniciar_Pesquisa_de_Mercado.bat`**, ou execute o **`Iniciar_App.bat`** criado na pasta do projeto.
2. Atrás dos panos, o script vai automaticamente ativar o `.venv` e executar um comando `streamlit run app_v2.py`.
3. Uma tela no seu navegador padrão abrirá automaticamente o Sistema!

> **Se preferir executar no modo Desenvolvedor (Terminal):**
> Visto que você possui as dependências instaladas, apenas rode `.\.venv\Scripts\activate` para ativar o ambiente virtual, seguido por `streamlit run app_v2.py` ou `python -m streamlit run app_v2.py`.

---

## 🔎 Fluxo em 3 Etapas (Como usar)

**Etapa 1: Extração Mercado (Google Shopping)**
- Você envia uma lista de Categoria/Palavras Chaves (ex: "Computador i7", "Teclado").
- O Engine puxa o cenário competitivo da SERP e aplica **Limpeza Tukey / IQR e Anti-Keywords** contidas no `regras_limpeza.json`.
- É traçada a Régua Estatística e salvos localmente os posicionamentos (Good, Better e Best ou "Econômico, Intermediário e Premium").

**Etapa 2: A Tabela Mestra (Input)**
- O usuário faz upload do escopo/planilha de precificação que possui atualmente (Onde estão informados "Seu Preço").

**Etapa 3: Cruzamento Oportunista Analytics**
- O sistema varre as pesquisas feitas, entende em que quartil o produto entra da concorrência e o reposiciona (Se for mais baixo a média de mercado para crescer marckup, se foi alto demais ele sinaliza fuga de Market Share).
- É formatado um Arquivo Completo de Exportação Excel C-Level.

---

## 📊 Integração ODS com Power BI

Toda pesquisa do passo 1 e os cruzamentos de portfólio do passo 3 geram *snapshots temporais* na base ODS `historico_pricing.db`. Ou seja: todo preço capturado mantém sua **data histórica real**.
- Uma barra lateral no Dashboard tem um link direto no Streamlit permitindo **"📥 Exportar Base Master PBI"**.
- Apenas um clique descarrega todos os recortes históricos purificados e formatados que podem integrar com relatórios BI de painel C-Level.

---

## 🛠️ Manutenção (Git & Atualizações)

Arquivos não rastreados que NUNCA devem ir ao Git:
- `.env` (Chaves privadas)
- `historico_pricing.db` (Banco local, é seu Master ODS particular, não deve ir ao respositório)
- Diretórios de cache como `__pycache__` ou a virtualenv `.venv`
