# Genera colab_evaluacion.ipynb incrustando los .py de esta carpeta. Ejecutar tras cambiarlos.
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CARPETA = Path(__file__).parent
ARCHIVOS = [
    "verificador.py", "instancia.py", "generador.py", "prompt.py", "modelo.py",
    "modelo_hf.py", "entorno.py", "evaluacion.py", "test_ambiente.py",
]

celdas = [
    new_markdown_cell(
        "# Evaluación de modelos en selección sobre árboles\n\n"
        "Activa la GPU en *Entorno de ejecución > Cambiar tipo de entorno*. Luego ejecuta todo, "
        "o edita primero la celda de configuración. Todos los modelos reciben los mismos árboles."
    ),
    new_code_cell("!pip install -q -U transformers accelerate mistral-common\n!nvidia-smi -L"),
    new_markdown_cell("## Código del ambiente"),
    new_code_cell("!mkdir -p seleccion_arbol"),
]
for nombre in ARCHIVOS:
    fuente = (CARPETA / nombre).read_text(encoding="utf-8")
    celdas.append(new_code_cell(f"%%writefile seleccion_arbol/{nombre}\n{fuente}"))

celdas += [
    new_code_cell(
        "import sys\n"
        "sys.path.insert(0, 'seleccion_arbol')\n"
        "!cd seleccion_arbol && python -m unittest test_ambiente 2>&1 | tail -3"
    ),
    new_markdown_cell(
        "## Configuración\n\n"
        "Cada entrada se evalúa por separado. `razonar` activa o desactiva el razonamiento en los "
        "modelos que lo permiten. Si a los 8192 tokens el razonamiento sigue abierto, se cierra y el "
        "modelo continúa hasta `tokens_maximos`. Los resultados van a Drive, así que al repetir la "
        "evaluación tras un corte de sesión se continúa donde quedó."
    ),
    new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n\n"
        "RUTA = '/content/drive/MyDrive/seleccion_arbol/resultados_con_tope.jsonl'\n"
        "NIVELES_A_EVALUAR = [2, 3, 4, 5]\n"
        "EPISODIOS = 5\n"
        "SEMILLA = 0\n\n"
        "MODELOS = {\n"
        "    'Qwen3.5 2B': dict(nombre='Qwen/Qwen3.5-2B', razonar=True, tokens_maximos=12000),\n"
        "    'Llama 3.2 1B': dict(nombre='unsloth/Llama-3.2-1B-Instruct', razonar=True, tokens_maximos=12000),\n"
        "    'OLMo 2 1B': dict(nombre='allenai/OLMo-2-0425-1B-Instruct', razonar=True, tokens_maximos=12000),\n"
        "    'DeepSeek R1 Distill 1.5B': dict(nombre='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', razonar=True, tokens_maximos=12000),\n"
        "    'Ministral 3 3B Reasoning': dict(nombre='mistralai/Ministral-3-3B-Reasoning-2512', razonar=True, tokens_maximos=12000),\n"
        "}"
    ),
    new_markdown_cell("## Evaluación"),
    new_code_cell(
        "from evaluacion import evaluar_modelo, resumir\n"
        "from modelo_hf import ModeloTransformers\n\n"
        "for nombre, opciones in MODELOS.items():\n"
        "    modelo = ModeloTransformers(**opciones)\n"
        "    try:\n"
        "        evaluar_modelo(modelo, nombre, NIVELES_A_EVALUAR, EPISODIOS, RUTA, SEMILLA)\n"
        "    except Exception as error:\n"
        "        # Un modelo que no carga no debe impedir evaluar los demás.\n"
        "        print(f'{nombre}: se omite por {type(error).__name__}: {error}')\n"
        "    finally:\n"
        "        modelo.liberar()"
    ),
    new_markdown_cell(
        "## Resultados\n\n"
        "Igualar la regla simple vale 0,5 de recompensa; una respuesta inválida vale -1."
    ),
    new_code_cell("resumen = resumir(RUTA)\nresumen"),
    new_code_cell(
        "ax = resumen['recompensa'].unstack('modelo').plot(marker='o', ylim=(-1.05, 1.05), ylabel='Recompensa media')\n"
        "ax.axhline(0.5, color='gray', linestyle='--', label='Regla simple')\n"
        "ax.legend();"
    ),
]

cuaderno = new_notebook(cells=celdas)
cuaderno.metadata["accelerator"] = "GPU"
cuaderno.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbformat.write(cuaderno, CARPETA / "colab_evaluacion.ipynb")
