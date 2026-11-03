# Genera colab_evaluacion.ipynb incrustando los .py de esta carpeta. Ejecutar tras cambiarlos.
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CARPETA = Path(__file__).parent
ARCHIVOS = [
    "verificador.py", "instancia.py", "generador.py", "prompt.py", "modelo.py",
    "modelo_hf.py", "entorno.py", "evaluacion.py", "dataset.py", "test_ambiente.py",
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
        "modelos que lo permiten. Si el modelo agota `tokens_pensamiento`, se cierra su razonamiento y "
        "debe responder con `tokens_respuesta`. Los resultados van a Drive, así que al repetir la "
        "evaluación tras un corte de sesión se continúa donde quedó."
    ),
    new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n\n"
        "RUTA = '/content/drive/MyDrive/seleccion_arbol/linea_base_test.jsonl'\n"
        "RUTA_DATOS = 'datos/test.jsonl'\n"
        "PRESUPUESTO = dict(tokens_pensamiento=3400, tokens_respuesta=200)\n"
        "NIVELES_A_EVALUAR = [2, 3, 4, 5, 6]\n"
        "INTENTOS = 3  # El k de pass@k\n\n"
        "MODELOS = {\n"
        "    # Qwen 2B, el modelo objetivo, se mide en LM Studio con baseline_lmstudio.ipynb.\n"
        "    # 'Qwen3.5 2B': dict(nombre='Qwen/Qwen3.5-2B', razonar=True, **PRESUPUESTO),\n"
        "    'Llama 3.2 1B': dict(nombre='unsloth/Llama-3.2-1B-Instruct', razonar=True, **PRESUPUESTO),\n"
        "    'OLMo 2 1B': dict(nombre='allenai/OLMo-2-0425-1B-Instruct', razonar=True, **PRESUPUESTO),\n"
        "    'DeepSeek R1 Distill 1.5B': dict(nombre='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', razonar=True, **PRESUPUESTO),\n"
        "    'Ministral 3 3B Reasoning': dict(nombre='mistralai/Ministral-3-3B-Reasoning-2512', razonar=True, **PRESUPUESTO),\n"
        "}"
    ),
    new_markdown_cell("## Evaluación"),
    new_code_cell(
        "from dataset import generar_datasets, guardar_datasets\n"
        "from evaluacion import evaluar_dataset, resumir, resumir_pass\n"
        "from modelo_hf import ModeloTransformers\n\n"
        "# Se generan los mismos árboles publicados en el repositorio.\n"
        "guardar_datasets(generar_datasets(), 'datos')\n\n"
        "for nombre, opciones in MODELOS.items():\n"
        "    modelo = ModeloTransformers(**opciones)\n"
        "    try:\n"
        "        evaluar_dataset(modelo, nombre, RUTA_DATOS, RUTA, niveles=NIVELES_A_EVALUAR, intentos=INTENTOS)\n"
        "    except Exception as error:\n"
        "        # Un modelo que no carga no debe impedir evaluar los demás.\n"
        "        print(f'{nombre}: se omite por {type(error).__name__}: {error}')\n"
        "    finally:\n"
        "        modelo.liberar()"
    ),
    new_markdown_cell(
        "## Resultados\n\n"
        "Igualar el mejor atajo vale 0,5 de recompensa; una respuesta inválida vale -1. "
        "`pass@1` son los intentos que logran el óptimo, y `pass@k` los árboles donde lo logra al menos uno de k intentos."
    ),
    new_code_cell("resumir_pass(RUTA, k=INTENTOS)"),
    new_code_cell("resumen = resumir(RUTA)\nresumen"),
    new_code_cell(
        "ax = resumen['recompensa'].unstack('modelo').plot(marker='o', ylim=(-1.05, 1.05), ylabel='Recompensa media')\n"
        "ax.axhline(0.5, color='gray', linestyle='--', label='Mejor atajo')\n"
        "ax.legend();"
    ),
]

cuaderno = new_notebook(cells=celdas)
cuaderno.metadata["accelerator"] = "GPU"
cuaderno.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbformat.write(cuaderno, CARPETA / "colab_evaluacion.ipynb")
