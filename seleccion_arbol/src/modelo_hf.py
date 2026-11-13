import gc

from src.modelo import ModeloLLM

CIERRE_RAZONAMIENTO = "\n</think>\n\n"


def razonamiento_abierto(prompt: str, texto: str) -> bool:
    return "</think>" not in texto and (prompt.rstrip().endswith("<think>") or "<think>" in texto)


def extraer_respuesta(prompt: str, texto: str) -> str:
    # Devuelve lo escrito después del razonamiento. Si este quedó sin cerrar, no hay respuesta.
    if "</think>" in texto:
        return texto.rsplit("</think>", 1)[1].strip()
    if razonamiento_abierto(prompt, texto):
        return ""
    return texto.strip()


class ModeloTransformers(ModeloLLM):
    # El modelo se carga en la primera consulta: si ya está evaluado, no ocupa la GPU.
    def __init__(
        self,
        nombre: str,
        razonar: bool | None = None,
        temperatura: float = 0.7,
        tokens_pensamiento: int = 8192,
        tokens_respuesta: int = 200,
    ):
        self.nombre = nombre
        self.razonar = razonar  # None usa lo que el modelo haga por defecto
        self.temperatura = temperatura
        self.tokens_pensamiento = tokens_pensamiento
        self.tokens_respuesta = tokens_respuesta
        self.ultimo_tokens = None
        self.ultimo_fin = None
        self.ultimo_texto = None
        self.ultimo_respuesta_forzada = None
        self._modelo = None
        self._tokenizador = None

    def _cargar(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self._tokenizador = AutoTokenizer.from_pretrained(self.nombre)
        try:
            self._modelo = AutoModelForImageTextToText.from_pretrained(
                self.nombre, torch_dtype="auto", device_map="auto"
            )
        except ValueError:
            self._modelo = AutoModelForCausalLM.from_pretrained(
                self.nombre, torch_dtype="auto", device_map="auto"
            )

    def generar(self, prompt: str) -> str:
        import torch

        if self._modelo is None:
            self._cargar()

        texto_prompt, entrada = self._preparar(prompt)
        entrada = entrada.to(self._modelo.device)
        largo_prompt = entrada["input_ids"].shape[1]
        self.ultimo_fin = "stop"
        self.ultimo_respuesta_forzada = False

        with torch.inference_mode():
            salida = self._continuar(entrada, self.tokens_pensamiento)
            texto = self._decodificar(salida, largo_prompt)

            # Terminar antes del límite significa que el modelo ya dejó de generar por sí mismo.
            if salida.shape[1] - largo_prompt == self.tokens_pensamiento and "</respuesta>" not in texto:
                self.ultimo_fin = "length"
                self.ultimo_respuesta_forzada = True
                cierre = CIERRE_RAZONAMIENTO if razonamiento_abierto(texto_prompt, texto) else "\n"
                forzado = self._tokenizador(
                    f"{cierre}<respuesta>", add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(salida.device)
                salida = torch.cat([salida, forzado], dim=1)
                largo_previo = salida.shape[1]
                salida = self._continuar(
                    {"input_ids": salida, "attention_mask": torch.ones_like(salida)},
                    self.tokens_respuesta,
                    stop_strings=["</respuesta>"],
                    tokenizer=self._tokenizador,
                )
                texto = self._decodificar(salida, largo_prompt)
                # Una lista cortada por el presupuesto no se cierra: podría parecer válida sin serlo.
                if "</respuesta>" not in texto and salida.shape[1] - largo_previo < self.tokens_respuesta:
                    texto += "</respuesta>"

        self.ultimo_tokens = salida.shape[1] - largo_prompt
        self.ultimo_texto = texto
        return extraer_respuesta(texto_prompt, texto)

    def _preparar(self, prompt: str):
        # Se tokeniza dentro de la plantilla. Tokenizar su texto aparte duplica el token de inicio (Llama)
        # y codifica como texto los tokens de control, como [INST] en Mistral.
        opciones = {} if self.razonar is None else {"enable_thinking": self.razonar}
        entrada = self._tokenizador.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **opciones,
        )
        # El texto solo se usa para saber si la plantilla abrió el razonamiento con <think>.
        texto_prompt = self._tokenizador.decode(entrada["input_ids"][0], skip_special_tokens=False)
        return texto_prompt, entrada

    def _continuar(self, entrada, tokens, **opciones):
        return self._modelo.generate(
            **entrada,
            max_new_tokens=tokens,
            do_sample=self.temperatura > 0,
            temperature=self.temperatura if self.temperatura > 0 else None,
            **opciones,
        )

    def _decodificar(self, salida, largo_prompt: int) -> str:
        return self._tokenizador.decode(salida[0][largo_prompt:], skip_special_tokens=True)

    def liberar(self):
        import torch

        self._modelo = None
        self._tokenizador = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
