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
Elige un conjunto de nodos de este árbol para obtener el mayor puntaje posible.

Valor de cada nodo:
1: 5
2: 6
3: 5

Conexiones:
1-2
2-3

Costo c = 2

Puntaje: cada nodo elegido aporta su valor menos c por cada vecino que también esté elegido. Los nodos no elegidos no aportan nada. Si no eliges ninguno, el puntaje es 0.

Termina con los números de los nodos elegidos, por ejemplo:
<respuesta>2 5 9</respuesta>
Si no eliges ninguno, escribe <respuesta></respuesta>.
```

## Características

El ambiente genera sus propios casos, conoce siempre la respuesta óptima y califica sin revisión humana.

### Episodios

Cada episodio tiene un solo paso, con la misma forma `reset` y `step` que usan otros ambientes de RL:

1. `reset(nivel)` crea un árbol y devuelve el prompt.
2. El modelo genera su respuesta.
3. `step(respuesta)` devuelve un `Resultado` con la recompensa, el puntaje, el puntaje de la regla simple, el óptimo y el error, si lo hubo.

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

Cada nodo se conecta con uno anterior elegido al azar, y luego se barajan los números para ocultar la raíz. El generador descarta un árbol si su óptimo no es positivo o si la regla simple rinde demasiado, con un máximo de 2000 intentos por árbol.

### Regla simple

La regla simple, llamada `voraz` en el código, elige todos los nodos con valor positivo, o ninguno si eso resta puntos. Sirve de referencia: un modelo que no la supera no está mostrando más que esa intuición.

El generador solo acepta árboles donde la regla logra como máximo el 70% del óptimo (`voraz_maximo` en `Nivel`). El camino del ejemplo anterior es un caso hecho a mano: allí la regla logra el 80%, así que el generador lo descartaría.

### Recompensa

Igualar la regla simple vale 0,5, y el resto solo se gana superándola. Así el modelo tiene una señal para mejorar incluso si no llega al óptimo.

| Respuesta | Recompensa | Cálculo |
| --- | --- | --- |
| Inválida | -1 | Fijo |
| Puntaje negativo o cero | 0 | Fijo |
| Bajo la regla simple | De 0 a 0,5 | 0,5 × puntaje / regla |
| Igual a la regla simple | 0,5 | Fijo, si la regla da más de 0 |
| Sobre la regla simple | De 0,5 a 1 | 0,5 + 0,5 × (puntaje - regla) / (óptimo - regla) |
| Óptima | 1 | Fijo |

### Formato de respuesta

El modelo puede razonar antes de responder, porque solo se lee el último bloque `<respuesta>...</respuesta>`. La respuesta es inválida en estos casos:

- No hay ningún bloque completo.
- El bloque contiene algo distinto de números separados por espacios o comas.
- Hay nodos repetidos.
- Algún nodo está fuera del rango de 1 a `n`.

### Verificación del óptimo

El óptimo viene de `calcular_optimo` en `verificador.py`, una programación dinámica sobre el árbol que decide en cada nodo entre elegirlo o no. Las pruebas la comparan con fuerza bruta en 1500 árboles de hasta 11 nodos.

### Modelos intercambiables

Cualquier modelo sirve si implementa `ModeloLLM.generar(prompt)`, que recibe el prompt y devuelve el texto de la respuesta.

| Clase | Archivo | Dónde corre |
| --- | --- | --- |
| `ModeloLMStudio` | `modelo.py` | En tu PC, con la API de LM Studio |
| `ModeloTransformers` | `modelo_hf.py` | En Colab o donde haya GPU, con Hugging Face |

`ModeloTransformers` agrega un tope de razonamiento: si a los 8192 tokens sigue abierto `<think>`, lo cierra y deja que el modelo continúe hasta 12000 tokens en total.

### Evaluación comparable

`evaluar_modelo` en `evaluacion.py` entrega los mismos árboles a todos los modelos, porque cada árbol depende solo de la semilla, el nivel y el episodio. Guarda cada episodio en un archivo JSONL al terminar y, si se interrumpe, retoma donde quedó.

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
python -m unittest test_ambiente
```

Deben pasar las 17 pruebas. Sin `numpy`, `test_mismos_arboles_y_reanuda` falla al llegar a `resumir`; las demás no lo necesitan.

### 2. Probar el ambiente sin un modelo

Esto escribe la respuesta a mano y sirve para ver cómo funciona. Ejecútalo desde `seleccion_arbol`:

```python
from entorno import EntornoSeleccion
from generador import NIVELES

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
from evaluacion import evaluar_modelo
from modelo import ModeloLMStudio

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
| `RUTA` | `.../seleccion_arbol/resultados_con_tope.jsonl` | Archivo de resultados en Drive |
| `MODELOS` | 5 modelos | Modelos que se comparan |

Los resultados quedan en Drive, así que puedes volver a ejecutar tras un corte de sesión. Con Qwen 2B, un episodio que agotó los tokens tardó entre 10 y 12 minutos.

Si cambias un archivo `.py`, regenera el notebook, porque lleva una copia de cada uno:

```bash
python generar_notebook.py
```

### 5. Usar otro modelo

Crea una clase que herede de `ModeloLLM`. Por ejemplo, así se juega un episodio:

```python
from entorno import EntornoSeleccion, jugar_episodio
from generador import NIVELES
from modelo import ModeloLLM

class MiModelo(ModeloLLM):
    def generar(self, prompt: str) -> str:
        return "<respuesta>1</respuesta>"  # Aquí va la llamada a tu modelo

print(jugar_episodio(EntornoSeleccion(semilla=1), MiModelo(), NIVELES[1]))
```

Esa misma clase funciona con `evaluar_modelo`.

## Cómo leer los resultados

La recompensa debe compararse con 0,5, que es lo que vale igualar la regla simple. Por encima de 0,5 el modelo la superó, y -1 significa que no entregó una respuesta válida.

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

El archivo JSONL guarda además `puntaje_voraz`, `error`, `respuesta` y, con `ModeloTransformers`, `texto_completo` y `razonamiento_cerrado`. Si tienes `pandas`, `resumir(ruta)` agrupa todo por modelo y nivel.

### Resultados hasta ahora

Qwen solo logró el óptimo hasta el nivel 3, y 7 de 21 respuestas fueron inválidas. En total hubo 3 óptimos, 2 respuestas sobre la regla sin llegar al óptimo y 9 bajo la regla.

Cada celda muestra la recompensa y, entre paréntesis, el puntaje frente al óptimo y a la regla simple. Corresponde a Qwen 2B (GGUF Q4_K_M) en LM Studio, con semilla 3, el 1 de noviembre de 2026.

| Nivel | Nodos | Episodio 1 | Episodio 2 | Episodio 3 |
| --- | --- | --- | --- | --- |
| 2 | 4 | 1,000 (16 de 16, regla 11) | 1,000 (11 de 11, regla 5) | -1,000 (eligió un nodo inexistente) |
| 3 | 5 | -1,000 (se cortó a 12000 tokens) | 0,312 (5 de 12, regla 8) | 1,000 (10 de 10, regla 6) |
| 4 | 7 | -1,000 (se cortó a 12000 tokens) | 0,543 (2 de 23, regla 0) | -1,000 (se cortó a 12000 tokens) |
| 5 | 10 | 0,438 (28 de 48, regla 32) | -1,000 (se cortó a 12000 tokens) | 0,083 (3 de 38, regla 18) |
| 6 | 15 | 0,981 (72 de 73, regla 47) | -1,000 (se cortó a 12000 tokens) | 0,219 (21 de 88, regla 48) |
| 7 | 25 | -1,000 (se cortó a 12000 tokens) | 0,375 (54 de 153, regla 72) | 0,109 (28 de 186, regla 128) |
| 8 | 40 | 0,485 (126 de 291, regla 130) | 0,072 (39 de 386, regla 270) | 0,000 (0 de 310, regla 162) |

Cada nivel tiene solo 3 episodios, así que estos resultados orientan pero no miden con precisión. De los 7 inválidos, 6 se deben a agotar los tokens y no a una mala selección. Por eso conviene reportar los cortes aparte de la recompensa media.

En Colab con los pesos originales, los 2 primeros episodios del nivel 2 se cortaron a los 12000 tokens. Eso ocurrió antes de agregar el tope de razonamiento, que aún no se ha probado con un modelo real.

## Limitaciones conocidas

El ambiente está probado, pero hay cuatro puntos que afectan cómo leer o usar sus resultados.

- **Salto en la recompensa:** si la regla simple da 0, un puntaje de 0 vale 0, pero cualquier puntaje positivo ya vale más de 0,5. En el nivel 4, un puntaje de 2 sobre 23 recibió 0,543.
- **Tokens agotados:** un corte vale -1, igual que un error de formato, así que la recompensa mezcla falta de tokens con falta de capacidad.
- **Diferencia entre LM Studio y Colab:** en local parece existir un tope cerca de 8192 tokens, sin confirmar. El de Colab lo imita, pero el modelo, la cuantización y el muestreo también son distintos, así que sus resultados no son directamente comparables.
- **Tope de razonamiento parcial:** solo actúa sobre modelos que usan la etiqueta `<think>`. No afecta a los que razonan sin ella o con otra etiqueta.

El ambiente tampoco incluye un ciclo de entrenamiento: entrega el prompt y la recompensa, y el algoritmo de RL queda por agregar.
