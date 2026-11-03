import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

# Formato de chat de Qwen. Otro modelo en LM Studio puede necesitar uno distinto.
PLANTILLA_QWEN = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


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
    ):
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

        if self.ultimo_fin != "length" or "</respuesta>" in contenido:
            return contenido
        return self._forzar_respuesta(prompt, previo)

    def _forzar_respuesta(self, prompt: str, previo: str) -> str:
        # Se usa texto sin plantilla de chat para que el modelo continúe justo después de <respuesta>.
        previo = previo.removeprefix("<think>").lstrip()
        cierre = "" if "</think>" in previo else "\n</think>\n\n"
        datos = self._consultar("completions", {
            "model": self.nombre,
            "prompt": f"{PLANTILLA_QWEN.format(prompt=prompt)}{previo}{cierre}<respuesta>",
            "temperature": self.temperatura,
            "max_tokens": self.tokens_respuesta,
            "stop": ["</respuesta>"],
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
