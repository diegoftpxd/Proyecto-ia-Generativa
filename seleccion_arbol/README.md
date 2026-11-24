# Ambiente de selección sobre árboles

## Resumen

El ambiente mide si un modelo de lenguaje sabe optimizar sobre un árbol, algo que una regla simple no resuelve. Crea árboles automáticamente en 8 niveles de dificultad y entrega cada uno al modelo como texto. Luego califica su respuesta con una recompensa entre -1 y 1, que se calcula sin intervención humana.

Está pensado para evaluar modelos y para entrenarlos con aprendizaje por refuerzo (RL) y currículum. El código está en esta carpeta, `seleccion_arbol`, del [repositorio](https://github.com/diegoftpxd/Proyecto-ia-Generativa).

## Qué prueba

La prueba es elegir el conjunto de nodos de un árbol que da el mayor puntaje posible. El modelo recibe el valor de cada nodo, las conexiones y un costo `c`, y responde con los nodos elegidos.

Cada nodo elegido aporta su valor menos `c` por cada vecino que también esté elegido. Los nodos no elegidos no aportan nada, y no elegir ninguno da 0. Por eso, cada conexión con ambos extremos elegidos resta `2c` en total.

### Qué se exige del modelo

- **Calcular bien:** aplicar la regla de puntaje a una selección concreta.
- **Pensar en conjunto:** el aporte de un nodo depende de qué otros nodos se eligen, así que no basta con mirarlos uno por uno.
- **Superar la intuición:** elegir todos los nodos positivos rara vez es lo mejor, y el ambiente lo garantiza (ver "Regla simple").
- **Terminar a tiempo y en formato:** debe entregar una respuesta legible antes de agotar sus tokens.

### Ejemplo

Un camino 1-2-3 con valores 5, 6 y 5, y costo `c = 2`:

| Selección | Cálculo | Puntaje | Recompensa |
| --- | --- | --- | --- |
| 1 y 3 (óptimo) | 5 + 5, sin vecinos elegidos | 10 | 1,000 |
| 1, 2 y 3 (regla simple) | 16 - 4 × 2 | 8 | 0,500 |
| 2 | 6 | 6 | 0,375 |
| Ninguno | | 0 | 0,000 |
| Texto sin bloque de respuesta | | | -1,000 |

Este es el prompt exacto que recibe el modelo para ese árbol:

```
Choose a set of nodes of the tree below to obtain the highest possible score.

The tree:
- It has 3 nodes, numbered from 1 to 3, and 2 edges. It is connected and has no cycles.
- Edges are bidirectional: "a-b" means that a is a neighbor of b and b is a neighbor of a.
- Each edge is listed exactly once. There are no duplicate edges, and no edge joins a node to itself.
- Node values are integers and can be positive, zero, or negative.
- The cost c is the same for every edge.

Node values (node: value):
1: 5
2: 6
3: 5

Edges:
1-2
2-3

Cost c = 2

Scoring rules:
- You may choose any set of nodes: none, some, or all of them. Two neighbors may both be chosen.
- Each chosen node contributes its value minus c for each of its neighbors that is also chosen.
- Only direct neighbors matter. Nodes further away do not change a node's contribution.
- So each edge with both of its nodes chosen lowers the total score by 2c, because each of the two nodes pays c.
- Nodes that are not chosen contribute 0 and pay no cost.
- The score is the sum of the contributions of the chosen nodes. Choosing no nodes scores 0.

Example with a different tree, only to show how the score works. It has 3 nodes with values 5, 6, and 5, edges 1-2 and 2-3, and cost c = 2:
- Choosing nodes 1 and 3 scores 5 + 5 = 10, because they are not neighbors.
- Choosing nodes 1, 2, and 3 scores (5 - 2) + (6 - 2 - 2) + (5 - 2) = 8.
- Choosing node 2 scores 6.
- The best choice is nodes 1 and 3, so the answer is <respuesta>1 3</respuesta>.

Answer format:
- End your reply with exactly one answer block that lists the chosen node numbers, separated by spaces.
- Use only node numbers from 1 to 3, each at most once. Do not write anything else inside the block.
- If the best choice is no nodes, write <respuesta></respuesta>.
```

## Características

El ambiente genera sus propios casos, conoce siempre la respuesta óptima y califica sin revisión humana.

### Episodios

Cada episodio tiene un solo paso, con la misma forma `reset` y `step` que usan otros ambientes de RL:

1. `reset(nivel)` crea un árbol y devuelve el prompt.
2. El modelo genera su respuesta.
3. `step(respuesta)` devuelve un `Resultado` con la recompensa, el puntaje, los puntajes de referencia, el óptimo y el error, si lo hubo.

### Interfaz común

`reward(prompt, completion) -> float`, en `src/entorno.py`, califica sin ningún estado guardado: reconstruye el árbol desde el prompt y lanza `ValueError` si el prompt no es de este ambiente.

```python
from src.entorno import reward

reward(prompt, "Razono...\n<respuesta>1 3</respuesta>")  # Devuelve un valor entre -1 y 1
```

Un intento pasa (pass) cuando su recompensa es 1, es decir, cuando logra el óptimo.

### Niveles

La dificultad sube con el tamaño del árbol, el rango de valores y el costo máximo. Los valores negativos aparecen desde el nivel 3.

| Nivel | Nodos | Valores | Costo máximo |
| --- | --- | --- | --- |
| 1 | 3 | 1 a 10 | 3 |
| 2 | 4 | 1 a 10 | 3 |
| 3 | 5 | -5 a 10 | 3 |
| 4 | 7 | -5 a 10 | 4 |
| 5 | 10 | -10 a 15 | 5 |
| 6 | 15 | -10 a 20 | 6 |
| 7 | 25 | -20 a 30 | 8 |
| 8 | 40 | -30 a 40 | 10 |

### Generación de árboles

Cada nodo se conecta con uno anterior elegido al azar. Luego se barajan los números, el orden de las conexiones y su sentido para ocultar la raíz. El generador descarta un árbol si su óptimo no es positivo o si un atajo rinde demasiado, con un máximo de 2000 intentos por árbol.

### Atajos

Hay dos estrategias que no optimizan y sirven de referencia:

- **Regla simple** (`voraz` en el código): elige todos los nodos positivos, o ninguno si eso resta puntos.
- **Sin vecinos:** toma los valores más altos sin elegir dos nodos conectados, así que nunca paga costos.

El generador exige que la regla simple logre como máximo el 70% del óptimo (`voraz_maximo` en `Nivel`). Desde el nivel 3, además, el mejor atajo no puede pasar del 90% (`atajo_maximo`), así que ningún atajo llega al óptimo. Los niveles 1 y 2 no tienen este segundo filtro a propósito: dejan aciertos alcanzables para que el modelo tenga señal de la que aprender.

El camino del ejemplo anterior es un caso hecho a mano: allí la regla simple logra el 80%, así que el generador lo descartaría.

### Recompensa

Igualar el mejor atajo vale 0,5, y el resto solo se gana superándolo. Así, seguir un atajo no da una recompensa alta, y el modelo tiene señal para mejorar aunque no llegue al óptimo.

| Respuesta | Recompensa | Cálculo |
| --- | --- | --- |
| Inválida | -1 | Fijo |
| Puntaje negativo o cero | 0 | Fijo |
| Bajo el mejor atajo | De 0 a 0,5 | 0,5 × puntaje / atajo |
| Igual al mejor atajo | 0,5 | Fijo, si el atajo no llega al óptimo |
| Sobre el mejor atajo | De 0,5 a 1 | 0,5 + 0,5 × (puntaje - atajo) / (óptimo - atajo) |
| Óptima | 1 | Fijo |
| Si un atajo ya logra el óptimo (niveles 1 y 2) | De 0 a 1 | puntaje / óptimo |

### Formato de respuesta

El prompt pide `<respuesta>...</respuesta>`, pero el verificador busca los nodos así:

1. **Entre `>` y `<`:** usa lo último escrito entre un `>` y el `<` siguiente que sea una lista de nodos o `ninguno`. Así acepta cierres mal escritos, como `<respuesta>1 3</ respuesta>`, otras etiquetas, como `<answer>1 3</answer>`, y el texto que el modelo agregue después. Un bloque `<respuesta></respuesta>` vacío es la selección vacía.
2. **Solo números:** si no hay nada entre `>` y `<`, acepta una respuesta que solo tenga la lista, como `1 3`.

El modelo puede razonar antes de responder, porque solo vale la última lista. Se aceptan formas equivalentes: cualquier orden, separadores como espacios, comas o punto y coma, `1 y 3`, `1 and 3`, `nodos 1, 3`, `{1, 3}` y `ninguno` para una selección vacía.

La respuesta es inválida en estos casos:

- No hay ninguna lista entre `>` y `<`, y la respuesta no es solo una lista.
- La lista contiene otra cosa, como alternativas (`1 3 o 2`), rangos (`1-3`), decimales o texto.
- Hay nodos repetidos.
- Algún nodo está fuera del rango de 1 a `n`.
- La lista supera los 2000 caracteres.
- La lista no se cierra con un `<`, como `<respuesta>1 3`, porque podría estar cortada.

Entregar varias listas no da lo mejor de cada una: solo vale la última. `PruebasReward`, en `src/test_ambiente.py`, reúne los casos diseñados para engañar al verificador.

### Datasets

`python -m src.dataset` genera `datos/entrenamiento.jsonl`, `datos/evaluacion.jsonl` y `datos/test.jsonl`. Tienen 800, 20 y 20 árboles por nivel (734 en el entrenamiento del nivel 1). Ningún árbol se repite entre particiones, ni siquiera cambiando la numeración de sus nodos.

Cada ejemplo trae `prompt`, `respuesta`, `razonamiento`, el árbol y sus puntajes de referencia. La respuesta y el razonamiento se calculan por código y se comprueban con el verificador. Si un árbol tiene varias soluciones óptimas, se guarda una: para calificar, usa `reward` en vez de comparar el texto.

`datos/test_extra.jsonl` es un test adicional que crece con el tiempo. Cada ronda del último bucle de `baseline_lmstudio.ipynb` le agrega un árbol nuevo por nivel, del 2 al 8, distinto de todos los de `datos`. Hoy tiene 84 árboles, 12 por nivel.

### Datos derivados: SFT y DPO

`python -m src.trazas resultados/04_trazas_500` muestrea un modelo profesor contra `datos/entrenamiento.jsonl` y genera los datos derivados que pide el desafío:

- `datos/sft.jsonl`: trazas correctas (`recompensa == 1.0`, sin corte y sin respuesta forzada), una por problema, en formato `messages`.
- `datos/dpo.jsonl`: pares `prompt`/`chosen`/`rejected` del mismo problema. El rechazo tiene recompensa entre 0.2 y 0.9: no es una respuesta inválida, sino una solución plausible pero peor.
- `datos/sft_solver.jsonl`: plus con las trazas que narra la programación dinámica de `src/dataset.py`. Son correctas por construcción, pero no reemplazan a las del modelo. La versión del repositorio sirve de punto de comparación: tiene los mismos 473 problemas de `datos/sft.jsonl`, en el mismo orden, así que un SFT con cada archivo usa los mismos problemas para entrenar y validar, y solo cambia el estilo de la traza.

El muestreo es secuencial hasta el primer éxito: por problema se generan hasta `muestras_por_problema` respuestas (6 por defecto) y se corta al obtener una correcta. Los intentos fallidos del camino se guardan y son los rechazos de DPO. Todo queda en `resultados/<carpeta>/muestras.jsonl`, así que `--solo-construir` rehace los datasets sin volver a llamar al modelo.

`configuracion.json` define el modelo, la temperatura, el presupuesto, la cuota por nivel y los umbrales. Antes de una corrida larga conviene probar:

```bash
python -m src.trazas resultados/04_trazas_500 --verificar
python -m src.trazas resultados/04_trazas_500 --niveles 2 --max-problemas 5
```

El resumen queda en `resultados/<carpeta>/generacion.csv`, con yield, cortes, inválidas y trazas por nivel. Si un nivel no alcanza la cuota, el archivo lo muestra: es preferible reportar el déficit a rellenar con trazas cortadas.

Los datos actuales vienen de `resultados/04_trazas_500`: Qwen 3.5 9B (GGUF Q4_K_M) en LM Studio, sin razonamiento oculto, temperatura 0,8 y 4096 tokens para pensar. El repositorio guarda su `configuracion.json` y su `generacion.csv`, pero no `muestras.jsonl`, así que `--solo-construir` necesita la copia local de esa corrida.

| Nivel | Problemas | Muestras | Yield | Cortes | Trazas SFT (cuota) | Pares DPO |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 251 | 355 | 70,4% | 1,4% | 250 (250) | 78 |
| 3 | 229 | 540 | 37,6% | 49,1% | 200 (200) | 37 |
| 4 | 126 | 694 | 3,7% | 95,4% | 23 (50) | 0 |
| Total | 606 | 1589 | 30,1% | 58,7% | 473 (500) | 115 (250) |

El nivel 4 quedó en déficit porque casi todas sus respuestas agotaron los tokens, no porque faltaran problemas. Para completarlo conviene subir `tokens_pensamiento` y el contexto de LM Studio antes que `muestras_por_problema`. Tampoco se alcanzó el objetivo de 250 pares DPO: hay pocos rechazos con recompensa entre 0,2 y 0,9.

### Verificación del óptimo

El óptimo viene de `calcular_optimo` en `src/verificador.py`, una programación dinámica sobre el árbol que decide en cada nodo entre elegirlo o no. Las pruebas la comparan con fuerza bruta en 1500 árboles de hasta 11 nodos.

### Modelos intercambiables

Cualquier modelo sirve si implementa `ModeloLLM.generar(prompt)`, que recibe el prompt y devuelve el texto de la respuesta.

| Clase | Archivo | Dónde corre |
| --- | --- | --- |
| `ModeloLMStudio` | `src/modelo.py` | En tu PC, con la API de LM Studio |
| `ModeloTransformers` | `src/modelo_hf.py` | En Colab o donde haya GPU, con Hugging Face |

Ambas clases usan dos presupuestos: `tokens_pensamiento` (8192 por defecto) y `tokens_respuesta` (200). Si el modelo agota el primero, se cierra su razonamiento, se escribe `<respuesta>` y debe completar los nodos con el segundo. Si esa lista también se corta, la respuesta es inválida. En LM Studio, la lista termina en el primer `<`, porque algunos modelos escriben mal el cierre, como `</ respuesta>`.

### Evaluación comparable

`evaluar_modelo` en `src/evaluacion.py` entrega los mismos árboles a todos los modelos, porque cada árbol depende solo de la semilla, el nivel y el episodio. Guarda cada episodio en un archivo JSONL al terminar y, si se interrumpe, retoma donde quedó.

## Cómo correrlo

Hay dos caminos con un modelo real: LM Studio en tu PC o el notebook en Colab. El ambiente en sí solo necesita Python.

### Requisitos

| Para qué | Qué necesitas |
| --- | --- |
| Ambiente y LM Studio | Python 3.10 o superior, sin paquetes adicionales (probado con 3.12) |
| `resumir` y todas las pruebas | `pandas` y `numpy` |
| Regenerar el notebook | `nbformat` |
| Colab | Una cuenta de Google y una GPU; el notebook instala el resto |

### 1. Descargar y probar

```bash
git clone https://github.com/diegoftpxd/Proyecto-ia-Generativa.git
cd Proyecto-ia-Generativa/seleccion_arbol
python -m unittest src.test_ambiente
```

Deben pasar las 63 pruebas. Se comprobó en el entorno de conda `tipico`, que tiene `pandas` y `numpy`. Sin ellos, las 9 pruebas de resúmenes fallan con `ImportError`.

### 2. Probar el ambiente sin un modelo

Esto escribe la respuesta a mano y sirve para ver cómo funciona. Ejecútalo desde `seleccion_arbol`:

```python
from src.entorno import EntornoSeleccion
from src.generador import NIVELES

entorno = EntornoSeleccion(semilla=1)
prompt = entorno.reset(NIVELES[1])  # NIVELES[1] es el nivel 2
print(prompt)

resultado = entorno.step("<respuesta>1 3</respuesta>")
print(resultado.recompensa, resultado.puntaje, resultado.puntaje_optimo)
```

### 3. Evaluar un modelo en tu PC con LM Studio

1. Abre LM Studio, carga el modelo e inicia el servidor local.
2. Revisa que el nombre y la dirección coincidan. Los valores por defecto son `qwen3.5-2b` y `http://localhost:1235/v1`.
3. Corre esto desde `seleccion_arbol`:

```python
from src.evaluacion import evaluar_modelo
from src.modelo import ModeloLMStudio

modelo = ModeloLMStudio()  # Otro modelo: ModeloLMStudio(nombre="...", url_base="...")
evaluar_modelo(modelo, "Qwen local", niveles=[2, 3, 4], episodios=3, ruta="resultados.jsonl")
```

En la RTX 2050 usada para las pruebas, cada episodio tardó entre 26 y 341 segundos. Mantén el PC encendido; si se corta, repite la llamada y continúa donde quedó.

### 4. Evaluar en Colab

1. Sube `seleccion_arbol/colab_evaluacion.ipynb` a Colab.
2. Activa la GPU en *Entorno de ejecución > Cambiar tipo de entorno*.
3. Edita la celda de configuración si quieres cambiar algo de la tabla siguiente.
4. Ejecuta todo y autoriza el acceso a Drive cuando lo pida.

| Variable | Valor actual | Qué controla |
| --- | --- | --- |
| `NIVELES_A_EVALUAR` | `[2, 3, 4, 5]` | Niveles que se evalúan |
| `EPISODIOS` | `5` | Episodios por nivel |
| `SEMILLA` | `0` | Árboles generados |
| `RUTA` | `.../seleccion_arbol/resultados_presupuesto.jsonl` | Archivo de resultados en Drive |
| `PRESUPUESTO` | `tokens_pensamiento=8192, tokens_respuesta=200` | Tokens para razonar y para responder |
| `MODELOS` | 5 modelos | Modelos que se comparan |

Los resultados quedan en Drive, así que puedes volver a ejecutar tras un corte de sesión. Con Qwen 2B, un episodio que agotó los tokens tardó entre 10 y 12 minutos.

Si cambias un archivo `.py`, regenera el notebook, porque lleva una copia de cada uno:

```bash
python -m src.generar_notebook
```

### 5. Usar otro modelo

Crea una clase que herede de `ModeloLLM`. Por ejemplo, así se juega un episodio:

```python
from src.entorno import EntornoSeleccion, jugar_episodio
from src.generador import NIVELES
from src.modelo import ModeloLLM

class MiModelo(ModeloLLM):
    def generar(self, prompt: str) -> str:
        return "<respuesta>1</respuesta>"  # Aquí va la llamada a tu modelo

print(jugar_episodio(EntornoSeleccion(semilla=1), MiModelo(), NIVELES[1]))
```

Esa misma clase funciona con `evaluar_modelo`.

## Cómo leer los resultados

La métrica principal es pass@k. **pass@1** es la proporción de intentos que logran el óptimo, y **pass@k** la probabilidad de que k intentos de un árbol incluyan al menos un óptimo. `resumir_pass(ruta, k)` las calcula por modelo y nivel. Para pass@k usa todos los intentos de cada árbol, con el estimador sin sesgo de Chen et al. (2021): con n intentos y c óptimos, `1 - C(n - c, k) / C(n, k)`. Los árboles con menos de k intentos quedan fuera.

La recompensa debe compararse con 0,5, que es lo que vale igualar el mejor atajo. Por encima de 0,5 el modelo lo superó, y -1 significa que no entregó una respuesta válida.

Este es un ejemplo de lo que imprime la evaluación por cada episodio:

```
Qwen3.5 2B nivel 2 ep 1: recompensa=-1.000 puntaje=None óptimo=15 697.9s 12000tok
```

| Campo | Qué significa |
| --- | --- |
| `recompensa` | Calificación del episodio, de -1 a 1 |
| `puntaje` | Puntaje de la selección; `None` si la respuesta es inválida |
| `óptimo` | Mejor puntaje posible en ese árbol |
| Segundos | Duración del episodio |
| Tokens | Tokens generados; igual al máximo si el modelo se cortó |

El archivo JSONL guarda además `puntaje_voraz`, `puntaje_sin_vecinos`, `error`, `fin`, `respuesta`, `texto_completo` y `respuesta_forzada`. Con `pandas`, `resumir(ruta)` agrupa todo por modelo y nivel.

Para el baseline en test, usa `baseline_lmstudio.ipynb` con el kernel del entorno de conda `tipico`. Evalúa en orden los modelos de `MODELOS`, en todos los niveles de `datos/test.jsonl`, con `evaluar_varios`. Si un modelo falla, sigue con el siguiente y lo reintenta al final.

### Resultados hasta ahora

Ningún modelo supera en promedio al mejor atajo: todas las recompensas medias quedan bajo 0,5. Ministral 3B es el mejor en pass@1 y en recompensa media, pero también entrega muchas respuestas inválidas. OLMo 1B tiene recompensa media negativa por sus respuestas inválidas.

Baseline en `datos/test.jsonl` (160 árboles, niveles 1 a 8):

| Modelo | Intentos | pass@1 | Inválidas | Recompensa media |
| --- | --- | --- | --- | --- |
| Ministral 3 3B Reasoning 4bit | 640 | 0,292 | 12,3% | 0,328 |
| DeepSeek R1 Distill 1.5B | 840 | 0,098 | 2,4% | 0,272 |
| Llama 3.2 1B | 840 | 0,057 | 0,0% | 0,238 |
| Qwen 3.5 2B | 640 | 0,059 | 4,8% | 0,178 |
| OLMo 2 1B | 840 | 0,058 | 23,8% | -0,034 |

Baseline en `datos/test_extra.jsonl` (niveles 2 a 8; no todos los modelos llegaron al mismo número de árboles):

| Modelo | Árboles | Intentos | pass@1 | Inválidas | Recompensa media |
| --- | --- | --- | --- | --- | --- |
| DeepSeek R1 Distill 1.5B | 84 | 336 | 0,012 | 1,2% | 0,212 |
| Llama 3.2 1B | 77 | 308 | 0,013 | 0,6% | 0,181 |
| Qwen 3.5 2B | 80 | 317 | 0,003 | 7,3% | 0,091 |
| OLMo 2 1B | 77 | 308 | 0,000 | 34,4% | -0,221 |
| Ministral 3 3B Reasoning 4bit | | | | | Sin corrida |

Las tablas mezclan todos los niveles, y los modelos no tienen el mismo número de intentos por árbol. En `test.jsonl`, Ministral y Qwen tienen 4 intentos por árbol. Los demás tienen 4 en los niveles 1, 7 y 8, y 6 en los niveles 2 a 6, porque una segunda corrida agregó los intentos 4 a 6. Para comparar con justicia, usa `resumir_pass(ruta, k)` por nivel.

Cada modelo tiene su carpeta en `BaseLines/`, con `baseline_test.jsonl`, `baseline_test_extra.jsonl` si se evaluó en el test extra y un `configuracion.json` que describe cada ejecución. Algunas carpetas guardan además `corrida.log` o el `prompt.py` usado. La columna `modelo` solo tiene el nombre del modelo. Todas las recompensas están calculadas con el verificador actual.

La primera prueba, con Qwen 2B en LM Studio el 1 de noviembre de 2026 (3 episodios por nivel), es anterior a los presupuestos de tokens, a la referencia del mejor atajo y al filtro desde el nivel 3, así que no se compara con estas tablas. En esa prueba, 6 de las 7 respuestas inválidas se debieron a agotar los 12000 tokens. Por eso conviene reportar los cortes aparte de la recompensa media.

## Limitaciones conocidas

El ambiente está probado, pero hay seis puntos que afectan cómo leer o usar sus resultados.

- **Otros atajos:** el filtro cubre dos estrategias conocidas. Un modelo podría encontrar otra regla que rinda bien sin optimizar; conviene revisar sus trazas al entrenar.
- **Niveles 1 y 2:** un atajo logra el óptimo en la mayoría de sus árboles, así que acertar ahí no demuestra que el modelo optimice.
- **Respuestas forzadas:** se califican igual que las demás, aunque el modelo no terminó de razonar. Por eso conviene mirar `cortes` junto con la recompensa.
- **Diferencia entre LM Studio y Colab:** el modelo, la cuantización y el muestreo son distintos, así que sus resultados no son directamente comparables.
- **Formato de chat en LM Studio:** la respuesta forzada usa el `formato` indicado a `ModeloLMStudio`, que por defecto es `QWEN`. `src/modelo.py` también trae `LLAMA`, `OLMO`, `DEEPSEEK` y `MINISTRAL`. Un formato incorrecto no da error, pero altera los resultados. `LLAMA` omite la fecha actual que agrega su plantilla original.
- **Etiquetas de razonamiento en Transformers:** el cierre forzado de `ModeloTransformers` solo reconoce `<think>`. Un modelo que razone con otra etiqueta recibe `<respuesta>`, pero su razonamiento no se cierra como tal.

El ambiente tampoco incluye un ciclo de entrenamiento: entrega el prompt y la recompensa, y el algoritmo de RL queda por agregar.
