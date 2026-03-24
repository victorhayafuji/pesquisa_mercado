@echo off
echo Iniciando a aplicacao Pesquisa de Mercado (app_v2.py)...
color 0A

:: Entra na pasta do projeto onde está o ambiente virtual e o código
cd /d "%~dp0pesquisa_mercado"

:: Ativa o ambiente virtual
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Aviso: Ambiente virtual ".venv" nao encontrado em %CD%
)

:: Executa a aplicacao Streamlit
echo Executando streamlit run app_v2.py...
streamlit run app_v2.py

:: Pausa em caso de erro ou fechamento manual para poder ler a tela
pause
