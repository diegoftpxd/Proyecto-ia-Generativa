import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class ModeloLLM(ABC):
    @abstractmethod
    def conversar(self, mensajes: list[dict], herramientas: list[dict]) -> dict:
        # Mensajes, herramientas y respuesta usan el formato de chat de OpenAI.
        pass


class ModeloLMStudio(ModeloLLM):
    def __init__(
        self,
        nombre: str = "qwen3.5-2b",
        url_base: str = "http://localhost:1235/v1",
        temperatura: float = 0.7,
        tokens_maximos: int = 1500,
        segundos_limite: float = 300,
    ):
        self.nombre = nombre
        self.url_base = url_base.rstrip("/")
        self.temperatura = temperatura
        self.tokens_maximos = tokens_maximos
        self.segundos_limite = segundos_limite

    def conversar(self, mensajes: list[dict], herramientas: list[dict]) -> dict:
        cuerpo = {
            "model": self.nombre,
            "messages": mensajes,
            "tools": herramientas,
            "tool_choice": "required",
            "temperature": self.temperatura,
            # Incluye el razonamiento oculto. Si se agota, el contenido puede llegar vacío.
            "max_tokens": self.tokens_maximos,
        }
        url = f"{self.url_base}/chat/completions"
        solicitud = urllib.request.Request(
            url,
            data=json.dumps(cuerpo).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(solicitud, timeout=self.segundos_limite) as respuesta:
                datos = json.load(respuesta)
        except urllib.error.HTTPError as error:
            detalle = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{url} respondió {error.code}: {detalle}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"No se pudo consultar {url}: {error}") from error

        try:
            mensaje = datos["choices"][0]["message"]
            # El razonamiento oculto no se devuelve para que no se acumule en la conversación.
            respuesta = {"role": "assistant", "content": mensaje.get("content") or ""}
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError(f"Respuesta inesperada de {url}: {datos}") from error

        if mensaje.get("tool_calls"):
            respuesta["tool_calls"] = mensaje["tool_calls"]
        return respuesta
