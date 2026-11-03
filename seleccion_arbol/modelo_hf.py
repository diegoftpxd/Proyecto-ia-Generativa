import gc

from modelo import ModeloLLM

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
        tokens_maximos: int = 12000,
        tokens_razonamiento: int = 8192,
    ):
        self.nombre = nombre
        self.razonar = razonar  # None usa lo que el modelo haga por defecto
        self.temperatura = temperatura
        self.tokens_maximos = tokens_maximos
        self.tokens_razonamiento = tokens_razonamiento  # Al llegar aquí, se cierra el razonamiento abierto
        self.ultimo_tokens = None
        self.ultimo_texto = None
        self.ultimo_razonamiento_cerrado = None
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

        opciones = {} if self.razonar is None else {"enable_thinking": self.razonar}
        texto_prompt = self._tokenizador.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **opciones,
        )
        entrada = self._tokenizador(texto_prompt, return_tensors="pt").to(self._modelo.device)
        largo_prompt = entrada["input_ids"].shape[1]
        limite = min(self.tokens_razonamiento, self.tokens_maximos)
        self.ultimo_razonamiento_cerrado = False

        with torch.inference_mode():
            salida = self._continuar(entrada, limite)

            # Terminar antes del límite significa que el modelo ya dejó de generar por sí mismo.
            if salida.shape[1] - largo_prompt == limite < self.tokens_maximos:
                if razonamiento_abierto(texto_prompt, self._decodificar(salida, largo_prompt)):
                    cierre = self._tokenizador(
                        CIERRE_RAZONAMIENTO, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].to(salida.device)
                    salida = torch.cat([salida, cierre], dim=1)
                    self.ultimo_razonamiento_cerrado = True
                restantes = self.tokens_maximos - (salida.shape[1] - largo_prompt)
                if restantes > 0:
                    salida = self._continuar(
                        {"input_ids": salida, "attention_mask": torch.ones_like(salida)}, restantes
                    )

        self.ultimo_tokens = salida.shape[1] - largo_prompt
        self.ultimo_texto = self._decodificar(salida, largo_prompt)
        return extraer_respuesta(texto_prompt, self.ultimo_texto)

    def _continuar(self, entrada, tokens):
        return self._modelo.generate(
            **entrada,
            max_new_tokens=tokens,
            do_sample=self.temperatura > 0,
            temperature=self.temperatura if self.temperatura > 0 else None,
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
