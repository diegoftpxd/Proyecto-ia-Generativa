import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class FormatoChat:
    # La plantilla solo se usa en la respuesta forzada, y omite el BOS porque lo agrega el tokenizador.
    plantilla: str
    apertura: str | None = None  # None: el modelo no tiene un canal de razonamiento
    cierre: str | None = None

    def abre_razonamiento(self) -> bool:
        return self.apertura is not None and self.plantilla.rstrip().endswith(self.apertura)

    def texto_visible(self, texto: str, cortado: bool) -> str:
        # Devuelve lo escrito después del razonamiento. Si quedó sin cerrar, no hay respuesta.
        if self.cierre is None:
            return texto
        cierre = self.cierre.strip()
        if cierre in texto:
            return texto.rsplit(cierre, 1)[1].strip()
        return "" if cortado and (self.abre_razonamiento() or self.apertura in texto) else texto


QWEN = FormatoChat(
    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n", "<think>", "\n</think>\n\n"
)
LLAMA = FormatoChat(
    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
OLMO = FormatoChat("<|user|>\n{prompt}\n<|assistant|>\n")
DEEPSEEK = FormatoChat("<｜User｜>{prompt}<｜Assistant｜><think>\n", "<think>", "\n</think>\n\n")
MINISTRAL = FormatoChat(
    "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a response. "
    "Format your response using Markdown, and use LaTeX for any mathematical equations. "
    "Write both your thoughts and the response in the same language as the input.\n\n"
    "Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, "
    "like working through an exercise on scratch paper. Be as casual and as long as you want "
    "until you are confident to generate the response to the user.[/THINK]"
    "Here, provide a self-contained response.[/SYSTEM_PROMPT][INST]{prompt}[/INST]",
    "[THINK]",
    "[/THINK]",
)


class ModeloLLM(ABC):
    @abstractmethod
    def generar(self, prompt: str) -> str:
        pass


class ModeloLMStudio(ModeloLLM):
    def __init__(
        self,
        nombre: str = "qwen3.5-2b",
        url_base: str = "http://localhost:1235/v1",
        temperatura: float = 0.7,
        tokens_pensamiento: int = 8192,
        tokens_respuesta: int = 200,
        segundos_limite: float = 600,
        formato: FormatoChat = QWEN,
    ):
        self.formato = formato
        self.nombre = nombre
        self.url_base = url_base.rstrip("/")
        self.temperatura = temperatura
        self.tokens_pensamiento = tokens_pensamiento
        self.tokens_respuesta = tokens_respuesta
        self.segundos_limite = segundos_limite
        self.ultimo_tokens = None
        self.ultimo_fin = None
        self.ultimo_texto = None
        self.ultimo_respuesta_forzada = None

    def _consultar(self, ruta: str, cuerpo: dict) -> dict:
        url = f"{self.url_base}/{ruta}"
        solicitud = urllib.request.Request(
            url,
            data=json.dumps(cuerpo).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(solicitud, timeout=self.segundos_limite) as respuesta:
                return json.load(respuesta)
        except urllib.error.HTTPError as error:
            detalle = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{url} respondió {error.code}: {detalle}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"No se pudo consultar {url}: {error}") from error

    def generar(self, prompt: str) -> str:
        datos = self._consultar("chat/completions", {
            "model": self.nombre,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperatura,
            # Incluye el razonamiento oculto.
            "max_tokens": self.tokens_pensamiento,
        })

        try:
            eleccion = datos["choices"][0]
            contenido = eleccion["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Respuesta inesperada de chat/completions: {datos}") from error

        self.ultimo_tokens = datos.get("usage", {}).get("completion_tokens")
        self.ultimo_fin = eleccion.get("finish_reason")
        self.ultimo_respuesta_forzada = False
        razonamiento = eleccion["message"].get("reasoning_content") or ""
        previo = f"{razonamiento}\n{contenido}".strip()
        self.ultimo_texto = previo

        # Con el razonamiento en su propio campo, el contenido ya es solo lo visible.
        cortado = self.ultimo_fin == "length"
        visible = contenido if razonamiento else self.formato.texto_visible(contenido, cortado)
        if not cortado or "</respuesta>" in visible:
            return visible
        return self._forzar_respuesta(prompt, previo)

    def _forzar_respuesta(self, prompt: str, previo: str) -> str:
        # Se usa texto sin plantilla de chat para que el modelo continúe justo después de <respuesta>.
        formato = self.formato
        cierre = "\n"
        if formato.apertura is not None:
            previo = previo.removeprefix(formato.apertura).lstrip()
            if formato.cierre.strip() not in previo:
                cierre = formato.cierre
            if not formato.abre_razonamiento():
                previo = f"{formato.apertura}{previo}"
        datos = self._consultar("completions", {
            "model": self.nombre,
            "prompt": f"{formato.plantilla.format(prompt=prompt)}{previo}{cierre}<respuesta>",
            "temperature": self.temperatura,
            "max_tokens": self.tokens_respuesta,
            # Termina en cualquier etiqueta: algunos modelos escriben mal el cierre, como "</ respuesta>".
            "stop": ["<"],
        })

        try:
            eleccion = datos["choices"][0]
            generado = eleccion["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Respuesta inesperada de completions: {datos}") from error

        # Una lista cortada por el presupuesto no se cierra: podría parecer válida sin serlo.
        final = "</respuesta>" if eleccion.get("finish_reason") != "length" else ""
        respuesta = f"<respuesta>{generado}{final}"
        self.ultimo_respuesta_forzada = True
        self.ultimo_tokens = (self.ultimo_tokens or 0) + datos.get("usage", {}).get("completion_tokens", 0)
        self.ultimo_texto = f"{previo}{cierre}{respuesta}"
        return respuesta
