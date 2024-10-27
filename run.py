#run.py

import os
from app.main import run_app

if __name__ == "__main__":
    # Se desejar, você pode definir variáveis de ambiente ou fazer outras configurações aqui
    os.environ['ENV'] = 'development'
    run_app()
