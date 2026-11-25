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


class HastaCierre:
    # Criterio de parada de generate: corta cada fila cuando lo generado desde `inicio` contiene `cierre`.
    # Reemplaza a stop_strings, que en transformers 5 usa métodos internos que el tokenizador de Mistral
    # (MistralCommonBackend) no tiene. Este solo necesita decode, que tiene cualquier tokenizador.
    def __init__(self, tokenizador, inicio: int, cierre: str = "</respuesta>"):
        self.tokenizador = tokenizador
        self.inicio = inicio
        self.cierre = cierre

    def __call__(self, input_ids, scores=None, **opciones):
        import torch

        terminadas = [
            self.cierre in self.tokenizador.decode(fila[self.inicio:], skip_special_tokens=True) for fila in input_ids
        ]
        return torch.tensor(terminadas, dtype=torch.bool, device=input_ids.device)


def tokenizar_prompt(tokenizador, prompt: str, razonar: bool | None, **opciones):
    # Se tokeniza dentro de la plantilla. Tokenizar su texto aparte duplica el token de inicio (Llama)
    # y codifica como texto los tokens de control, como [INST] en Mistral. razonar=None usa lo que el
    # modelo haga por defecto. entrenar_grpo.py la usa para que el entrenamiento vea el prompt de la evaluación.
    if razonar is not None:
        opciones["enable_thinking"] = razonar
    return tokenizador.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **opciones,
    )


class ModeloTransformers(ModeloLLM):
    # El modelo se carga en la primera consulta: si ya está evaluado, no ocupa la GPU.
    def __init__(
        self,
        nombre: str,
        razonar: bool | None = None,
        temperatura: float = 0.7,
        tokens_pensamiento: int = 8192,
        tokens_respuesta: int = 200,
        adaptador: str | list[str] | None = None,
    ):
        self.nombre = nombre
        # Carpetas de adaptadores LoRA, como los de entrenar_sft.py y entrenar_grpo.py. Se fusionan en orden:
        # el de GRPO se entrenó sobre el modelo con el SFT ya fusionado, así que va después de él.
        self.adaptadores = [adaptador] if isinstance(adaptador, str) else list(adaptador or [])
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
        if self.adaptadores:
            from peft import PeftModel

            # Se fusionan con los pesos: el modelo queda igual que el base y genera a la misma velocidad.
            for adaptador in self.adaptadores:
                self._modelo = PeftModel.from_pretrained(self._modelo, adaptador).merge_and_unload()

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
                    stopping_criteria=[HastaCierre(self._tokenizador, largo_previo)],
                )
                texto = self._decodificar(salida, largo_prompt)
                # Una lista cortada por el presupuesto no se cierra: podría parecer válida sin serlo.
                if "</respuesta>" not in texto and salida.shape[1] - largo_previo < self.tokens_respuesta:
                    texto += "</respuesta>"

        self.ultimo_tokens = salida.shape[1] - largo_prompt
        self.ultimo_texto = texto
        return extraer_respuesta(texto_prompt, texto)

    def _preparar(self, prompt: str):
        entrada = tokenizar_prompt(self._tokenizador, prompt, self.razonar, return_tensors="pt")
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
