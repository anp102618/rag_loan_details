import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [

    ".env",
    ".gitignore",
    "main.py",

    "Data/__init__.py",


    "src/__init__.py",
    "src/run.py",
    "src/config.py",

    "src/db/__init__.py",
    "src/db/main.py",
    #"src/db/session.py",

    "src/Users/__init__.py",
    "src/Users/models.py",
    "src/Users/routes.py",
    "src/Users/helpers.py",
    "src/Users/schemas.py",
    "src/Users/auth.py",

    "src/Utils/__init__.py",
    "src/Utils/logger_setup.py",
    "src/Utils/exception_handler.py",
    

    "src/RAG/__init__.py",
    "src/RAG/models.py",
    "src/RAG/routes.py",
    "src/RAG/app_state.py",
    "src/RAG/Strategies/__init__.py",
    "src/RAG/Utils/__init__.py",
    "src/RAG/Utils/circuit_breaker.py",
    "src/RAG/Utils/rate_limiter.py",
    "src/RAG/Utils/caching.py",
    "src/RAG/Utils/memory.py",
    
    "src/RAG/Service/__init__.py",
    "src/RAG/Service/extractor.py",
    "src/RAG/Service/chunker.py",
    "src/RAG/Service/normalizer.py",
    "src/RAG/Service/embedder.py",
    "src/RAG/Service/retriever.py",
    "src/RAG/Service/generator.py",
    "src/RAG/Service/pipeline.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")