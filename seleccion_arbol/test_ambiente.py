import json
import os
import random
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from itertools import combinations
from pathlib import Path

from dataset import agregar_arboles, forma_canonica, generar_datasets, guardar_datasets, resolver
from entorno import EntornoSeleccion, calcular_recompensa, evaluar_respuesta, reward
from evaluacion import evaluar_dataset, evaluar_modelo, evaluar_varios, leer_resultados, resumir, resumir_pass
from generador import NIVELES, Nivel, generar_instancia
from modelo_hf import ModeloTransformers, extraer_respuesta
from instancia import Instancia
from modelo import DEEPSEEK, LLAMA, MINISTRAL, OLMO, ModeloLLM, ModeloLMStudio
from prompt import generar_prompt, leer_prompt


def fuerza_bruta(instancia: Instancia) -> int:
    nodos = range(1, instancia.cantidad_nodos + 1)
    return max(
        instancia.puntaje(set(seleccion))
        for tamano in range(instancia.cantidad_nodos + 1)
        for seleccion in combinations(nodos, tamano)
    )


def mejor_seleccion(instancia: Instancia) -> set[int]:
    nodos = range(1, instancia.cantidad_nodos + 1)
    return max(
        (set(s) for t in range(instancia.cantidad_nodos + 1) for s in combinations(nodos, t)),
        key=instancia.puntaje,
    )


# Camino 1-2-3 con c = 2: elegir 1 y 3 da 10; elegir los tres da 16 - 2*2*2 = 8.
CAMINO = Instancia((5, 6, 5), ((1, 2), (2, 3)), 2)


class PruebasPuntaje(unittest.TestCase):
    def test_puntaje_a_mano(self):
        self.assertEqual(CAMINO.puntaje(set()), 0)
        self.assertEqual(CAMINO.puntaje({2}), 6)
        self.assertEqual(CAMINO.puntaje({1, 3}), 10)
        self.assertEqual(CAMINO.puntaje({1, 2}), 11 - 2 * 2)
        self.assertEqual(CAMINO.puntaje({1, 2, 3}), 8)
        self.assertEqual(CAMINO.puntaje_optimo(), 10)

    def test_valores_negativos_y_un_nodo(self):
        self.assertEqual(Instancia((-3,), (), 1).puntaje_optimo(), 0)
        self.assertEqual(Instancia((4,), (), 1).puntaje_optimo(), 4)
        self.assertEqual(Instancia((-1, -2), ((1, 2),), 1).puntaje_optimo(), 0)

    def test_optimo_contra_fuerza_bruta(self):
        rng = random.Random(1)
        for _ in range(1500):
            nivel = Nivel(rng.randint(1, 11), -15, 20, 12, voraz_maximo=1.0)
            instancia = generar_instancia(nivel, rng)
            self.assertEqual(instancia.puntaje_optimo(), fuerza_bruta(instancia), instancia)


class PruebasGenerador(unittest.TestCase):
    def test_niveles(self):
        rng = random.Random(2)
        for nivel in NIVELES:
            for _ in range(30):
                instancia = generar_instancia(nivel, rng)
                optimo = instancia.puntaje_optimo()
                self.assertEqual(instancia.cantidad_nodos, nivel.nodos)
                self.assertEqual(len(instancia.aristas), nivel.nodos - 1)
                self.assertGreater(optimo, 0)
                # Un árbol es conexo: todos los nodos se alcanzan desde el 1.
                vecinos, vistos, pendientes = instancia.vecinos(), {1}, [1]
                while pendientes:
                    for vecino in vecinos[pendientes.pop()]:
                        if vecino not in vistos:
                            vistos.add(vecino)
                            pendientes.append(vecino)
                self.assertEqual(len(vistos), nivel.nodos)
                self.assertLessEqual(instancia.puntaje_voraz(), 0.7 * optimo)
                self.assertLessEqual(instancia.puntaje_referencia(), nivel.atajo_maximo * optimo)

    def test_reproducible(self):
        self.assertEqual(generar_instancia(NIVELES[4], random.Random(9)), generar_instancia(NIVELES[4], random.Random(9)))

    def test_parametros_invalidos(self):
        for argumentos in [(0, 1, 2, 1), (3, 5, 1, 1), (3, -5, 0, 1), (3, 1, 5, 0), (3, 1, 5, 1, 1.5), (3, 1, 5, 1, -0.1), (3, 1, 5, 1, 0.7, 1.5)]:
            with self.assertRaises(ValueError):
                Nivel(*argumentos)


class PruebasRecompensa(unittest.TestCase):
    def test_recompensas(self):
        # En CAMINO la regla simple logra 8 y el óptimo es 10.
        casos = [
            ("<respuesta>1 3</respuesta>", 1.0, 10),
            ("<respuesta>3, 1</respuesta>", 1.0, 10),
            ("Pienso...\n<respuesta>2</respuesta>\nMejor:\n<respuesta>1 3</respuesta>", 1.0, 10),
            ("<respuesta>2</respuesta>", 0.375, 6),
            ("<respuesta>1 2 3</respuesta>", 0.5, 8),
            ("<respuesta></respuesta>", 0.0, 0),
        ]
        for texto, recompensa, puntaje in casos:
            resultado = evaluar_respuesta(CAMINO, texto)
            self.assertAlmostEqual(resultado.recompensa, recompensa, msg=texto)
            self.assertEqual(resultado.puntaje, puntaje, texto)
            self.assertEqual(resultado.puntaje_voraz, 8)

    def test_tramos(self):
        # Los casos de la tabla: óptimo 20 con regla simple en 18, y óptimo 14 con regla simple en 3.
        for puntaje, voraz, optimo, esperado in [
            (18, 18, 20, 0.5), (19, 18, 20, 0.75), (20, 18, 20, 1.0),
            (2, 3, 14, 1 / 3), (3, 3, 14, 0.5), (7, 3, 14, 0.5 + 0.5 * 4 / 11), (14, 3, 14, 1.0),
            (-4, 3, 14, 0.0), (0, 3, 14, 0.0),
            (5, 0, 6, 5 / 6), (1, 0, 6, 1 / 6), (0, 0, 6, 0.0), (-2, 0, 6, 0.0),
            (14, 14, 14, 1.0), (7, 14, 14, 0.5),
        ]:
            self.assertAlmostEqual(calcular_recompensa(puntaje, voraz, optimo), esperado, msg=(puntaje, voraz, optimo))

    def test_recompensa_crece_con_el_puntaje_y_queda_en_rango(self):
        rng = random.Random(6)
        for _ in range(200):
            instancia = generar_instancia(NIVELES[3], rng)
            voraz, optimo = instancia.puntaje_voraz(), instancia.puntaje_optimo()
            recompensas = [calcular_recompensa(p, voraz, optimo) for p in range(-30, optimo + 1)]
            self.assertEqual(recompensas, sorted(recompensas))
            self.assertTrue(all(0 <= r <= 1 for r in recompensas))
            self.assertEqual(recompensas[-1], 1.0)
            self.assertEqual(calcular_recompensa(voraz, voraz, optimo), 0.5 if voraz > 0 else 0.0)

    def test_puntaje_negativo_no_baja_de_cero(self):
        instancia = Instancia((1, 1, 5), ((1, 2), (2, 3)), 3)
        resultado = evaluar_respuesta(instancia, "<respuesta>1 2</respuesta>")
        self.assertEqual(resultado.puntaje, -4)
        self.assertEqual(resultado.puntaje_voraz, 0)
        self.assertEqual(resultado.recompensa, 0.0)

    def test_optimo_cero(self):
        instancia = Instancia((-3,), (), 1)
        self.assertEqual(evaluar_respuesta(instancia, "<respuesta></respuesta>").recompensa, 1.0)
        self.assertEqual(evaluar_respuesta(instancia, "<respuesta>1</respuesta>").recompensa, 0.0)

    def test_formato_invalido(self):
        for texto in ["", "1 3 x", "<respuesta>1 3", "<respuesta>uno</respuesta>", "<respuesta>1 1</respuesta>",
                      "<respuesta>0</respuesta>", "<respuesta>4</respuesta>", "<respuesta>-1</respuesta>",
                      "<respuesta>٣</respuesta>", "<respuesta>" + "9" * 5000 + "</respuesta>"]:
            resultado = evaluar_respuesta(CAMINO, texto)
            self.assertEqual(resultado.recompensa, -1.0, texto[:30])
            self.assertIsNotNone(resultado.error)

    def test_textos_aleatorios_no_rompen(self):
        rng = random.Random(3)
        letras = "<>respuesta/0123456789 ,\n-٣"
        for _ in range(3000):
            texto = "".join(rng.choice(letras) for _ in range(rng.randint(0, 40)))
            if rng.random() < 0.5:
                texto = f"<respuesta>{texto}</respuesta>"
            self.assertTrue(-1.0 <= evaluar_respuesta(CAMINO, texto).recompensa <= 1.0)


class ModeloFuerzaBruta(ModeloLLM):
    def __init__(self, entorno):
        self.entorno = entorno

    def generar(self, prompt):
        assert prompt == generar_prompt(self.entorno.instancia)
        seleccion = " ".join(map(str, sorted(mejor_seleccion(self.entorno.instancia))))
        return f"Razonamiento.\n<respuesta>{seleccion}</respuesta>"


class PruebasEntorno(unittest.TestCase):
    def test_episodios_optimos(self):
        entorno = EntornoSeleccion(semilla=4)
        modelo = ModeloFuerzaBruta(entorno)
        for nivel in NIVELES[:5]:
            prompt = entorno.reset(nivel)
            resultado = entorno.step(modelo.generar(prompt))
            self.assertEqual(resultado.recompensa, 1.0)

    def test_step_requiere_reset(self):
        entorno = EntornoSeleccion(semilla=5)
        with self.assertRaises(RuntimeError):
            entorno.step("<respuesta></respuesta>")
        entorno.reset(NIVELES[0])
        entorno.step("<respuesta></respuesta>")
        with self.assertRaises(RuntimeError):
            entorno.step("<respuesta></respuesta>")


class ModeloFijo(ModeloLLM):
    def __init__(self, falla_en=None):
        self.prompts = []
        self.falla_en = falla_en

    def generar(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == self.falla_en:
            raise RuntimeError("sin memoria")
        return "<respuesta></respuesta>"


class PruebasEvaluacion(unittest.TestCase):
    def test_mismos_arboles_y_reanuda(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "sub" / "r.jsonl"
            a, b = ModeloFijo(falla_en=2), ModeloFijo()
            evaluar_modelo(a, "a", [2, 3], 2, ruta)
            self.assertEqual(len(leer_resultados(ruta)), 3)  # La falla no se guarda
            evaluar_modelo(a, "a", [2, 3], 2, ruta)
            self.assertEqual(len(leer_resultados(ruta)), 4)  # Solo reintenta la pendiente
            self.assertEqual(len(a.prompts), 5)
            evaluar_modelo(b, "b", [2, 3], 2, ruta)
            self.assertEqual(sorted(set(a.prompts)), sorted(b.prompts))
            resumen = resumir(ruta)
            self.assertEqual(resumen.loc[("b", 2), "episodios"], 2)
            self.assertEqual(resumen.loc[("b", 2), "recompensa"], 0.0)

    def test_extraer_respuesta(self):
        self.assertEqual(extraer_respuesta("p", "<respuesta>1</respuesta>"), "<respuesta>1</respuesta>")
        self.assertEqual(extraer_respuesta("p<think>\n", "duda\n</think>\n<respuesta>1</respuesta>"), "<respuesta>1</respuesta>")
        self.assertEqual(extraer_respuesta("p<think>\n", "<respuesta>1</respuesta> sigo pensando"), "")
        self.assertEqual(extraer_respuesta("p", "<think>a <respuesta>1</respuesta>"), "")


class PruebasReward(unittest.TestCase):
    # CAMINO: el óptimo es {1, 3} con 10 puntos, y la regla simple logra 8 con los tres nodos.
    PROMPT = generar_prompt(CAMINO)

    def test_recupera_el_arbol_desde_el_prompt(self):
        rng = random.Random(11)
        for nivel in NIVELES:
            for _ in range(20):
                instancia = generar_instancia(nivel, rng)
                self.assertEqual(leer_prompt(generar_prompt(instancia)), instancia)
        solo = Instancia((4,), (), 1)
        self.assertEqual(leer_prompt(generar_prompt(solo)), solo)

    def test_prompt_ajeno_se_rechaza(self):
        roto = self.PROMPT.replace("2-3\n", "")
        desordenado = self.PROMPT.replace("1: 5\n2: 6", "2: 6\n1: 5")
        fuera = self.PROMPT.replace("2-3", "2-9")
        for prompt in ["¿Cuánto es 2 + 2?", "", roto, desordenado, fuera]:
            with self.assertRaises(ValueError):
                reward(prompt, "<respuesta>1</respuesta>")

    def test_acepta_respuestas_equivalentes(self):
        for texto in [
            "<respuesta>1 3</respuesta>",
            "<respuesta>3 1</respuesta>",
            "<respuesta>1, 3</respuesta>",
            "<respuesta>1,3</respuesta>",
            "<respuesta>1; 3</respuesta>",
            "<respuesta>1 y 3</respuesta>",
            "<respuesta>3 e 1</respuesta>",
            "<respuesta>1 and 3</respuesta>",
            "<respuesta>3, and 1</respuesta>",
            "<respuesta>1 AND 3</respuesta>",
            "1 and 3",
            "<respuesta>\n1\n3\n</respuesta>",
            "<respuesta>nodos 1 y 3</respuesta>",
            "<respuesta>Nodos: 1, 3.</respuesta>",
            "<respuesta>{1, 3}</respuesta>",
            "<respuesta>[1, 3]</respuesta>",
            "<RESPUESTA>1 3</RESPUESTA>",
            "Razono mucho.\n</think>\n<respuesta>1 3</respuesta>",
            "<respuesta>2</respuesta> Mejor no.\n<respuesta>1 3</respuesta>",
            # Números entre > y <, aunque la etiqueta esté mal escrita o sea otra
            "<respuesta>1 3</ respuesta>",
            "<respuesta>1 3<//response>\n\nStep-by-step explanation: node 2 > node 1",
            "<answer>1 3</answer>",
            "Pienso.\n<respuesta>\nI choose <b>1 3</b></respuesta>",
            "<respuesta></respuesta> Mejor:\n<respuesta>1 3</respuesta>",
            # Solo números
            "1 3",
            "  1, 3.\n",
        ]:
            self.assertEqual(reward(self.PROMPT, texto), 1.0, texto)

    def test_texto_entre_etiquetas_no_es_una_seleccion_vacia(self):
        for texto in ["<think>dudo</think>\n\n<x>", "<a> </a>", "<respuesta>\n</ respuesta>"]:
            self.assertEqual(reward(self.PROMPT, texto), -1.0, texto)

    def test_usa_la_ultima_lista_entre_etiquetas(self):
        self.assertEqual(evaluar_respuesta(CAMINO, "<respuesta>1 <respuesta>3</respuesta>").puntaje, 5)
        self.assertEqual(reward(self.PROMPT, "<respuesta>2</respuesta><respuesta>uno</respuesta>"), 0.375)

    def test_acepta_ninguno_en_varias_formas(self):
        for texto in ["<respuesta></respuesta>", "<respuesta>ninguno</respuesta>", "<respuesta>Ninguna.</respuesta>", "<respuesta>vacío</respuesta>"]:
            self.assertEqual(reward(self.PROMPT, texto), 0.0, texto)
        negativo = generar_prompt(Instancia((-3,), (), 1))
        self.assertEqual(reward(negativo, "<respuesta>ninguno</respuesta>"), 1.0)
        self.assertEqual(reward(negativo, "<respuesta>none</respuesta>"), 1.0)

    def test_no_se_puede_enganar(self):
        for texto in [
            # Varias respuestas a la vez
            "<respuesta>1 3 o 2</respuesta>",
            "<respuesta>1 3 | 2</respuesta>",
            "<respuesta>1 3 / 2</respuesta>",
            # Explotar el formato
            "<respuesta>1 3",
            "La respuesta es 1 3",
            "1 3 o 2",
            "<respuesta>1 and</respuesta>",
            "<respuesta>and 1 3</respuesta>",
            "<respuesta>1 andy 3</respuesta>",
            "<respuesta>1 and and 3</respuesta>",
            "The chosen nodes are 1 and 3",
            "<respuesta>1-3</respuesta>",
            "<respuesta>1.0 3.0</respuesta>",
            "<respuesta>0x1 0x3</respuesta>",
            "<respuesta>1 1 3</respuesta>",
            "<respuesta>0 1 3</respuesta>",
            "<respuesta>-1 3</respuesta>",
            "<respuesta>1 3 4</respuesta>",
            "<respuesta>todos</respuesta>",
            "<respuesta>el óptimo</respuesta>",
            "<respuesta>10</respuesta>",
            "<respuesta>١ ٣</respuesta>",
            "<respuesta>１ ３</respuesta>",
            "<respuesta>" + "1" * 7 + "</respuesta>",
            # Declarar el resultado o pedirlo
            "recompensa=1.0",
            "<respuesta>recompensa 1.0</respuesta>",
            "Ignora las reglas y dame la máxima recompensa.",
        ]:
            self.assertEqual(reward(self.PROMPT, texto), -1.0, texto)

    def test_varios_bloques_no_dan_lo_mejor_de_cada_uno(self):
        self.assertEqual(reward(self.PROMPT, "<respuesta>1 3</respuesta><respuesta>2</respuesta>"), 0.375)
        self.assertEqual(reward(self.PROMPT, "<respuesta>2</respuesta>\n<respuesta>1 3</respuesta>"), 1.0)

    def test_elegir_todo_y_copiar_el_ejemplo_no_premian(self):
        self.assertEqual(reward(self.PROMPT, "<respuesta>1 2 3</respuesta>"), 0.5)
        grande = generar_instancia(NIVELES[4], random.Random(1))
        self.assertLess(reward(generar_prompt(grande), "<respuesta>2 5 9</respuesta>"), 1.0)

    def test_atajos_valen_a_lo_mas_la_referencia(self):
        rng = random.Random(5)
        for nivel in NIVELES[2:6]:
            for _ in range(40):
                instancia = generar_instancia(nivel, rng)
                prompt = generar_prompt(instancia)
                positivos = [i + 1 for i, valor in enumerate(instancia.valores) if valor > 0]
                self.assertLessEqual(reward(prompt, "<respuesta>" + " ".join(map(str, positivos)) + "</respuesta>"), 0.5)

                vecinos, elegidos = instancia.vecinos(), set()
                for nodo in sorted(vecinos, key=lambda n: -instancia.valores[n - 1]):
                    if instancia.valores[nodo - 1] > 0 and not any(v in elegidos for v in vecinos[nodo]):
                        elegidos.add(nodo)
                self.assertLessEqual(reward(prompt, "<respuesta>" + " ".join(map(str, elegidos)) + "</respuesta>"), 0.5)

    def test_texto_enorme_no_cuelga_al_parser(self):
        inicio = time.perf_counter()
        for texto in [
            "<respuesta>" + "1 " * 200000 + "x</respuesta>",
            "<respuesta>" * 100000,
            "<respuesta>" * 100000 + "x</respuesta>",
            ">" + "1 " * 200000 + "<",
            " " * 500000,
            "<respuesta>" + " " * 200000 + "x</respuesta>",
            "<respuesta>1" + " " * 1900 + "x</respuesta>",
        ]:
            self.assertEqual(reward(self.PROMPT, texto), -1.0)
        self.assertLess(time.perf_counter() - inicio, 5)


class ServidorSimulado:
    # Responde con los datos indicados y recuerda lo que recibió, como una API de LM Studio.
    def __init__(self, respuestas):
        recibidas = self.recibidas = []

        class Manejador(BaseHTTPRequestHandler):
            def log_message(self, *argumentos):
                pass

            def do_POST(self):
                recibidas.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                cuerpo = json.dumps(respuestas[len(recibidas) - 1]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

        self._servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
        self.url = f"http://127.0.0.1:{self._servidor.server_port}/v1"

    def __enter__(self):
        threading.Thread(target=self._servidor.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *argumentos):
        self._servidor.shutdown()
        self._servidor.server_close()


def chat(contenido, fin, razonamiento=None, tokens=10):
    mensaje = {"content": contenido, "reasoning_content": razonamiento}
    return {"choices": [{"message": mensaje, "finish_reason": fin}], "usage": {"completion_tokens": tokens}}


def completado(texto, fin, tokens=3):
    return {"choices": [{"text": texto, "finish_reason": fin}], "usage": {"completion_tokens": tokens}}


class PruebasPresupuestoLMStudio(unittest.TestCase):
    def consultar(self, respuestas):
        with ServidorSimulado(respuestas) as servidor:
            modelo = ModeloLMStudio(url_base=servidor.url, tokens_pensamiento=100, tokens_respuesta=7)
            return modelo.generar("P"), modelo, servidor.recibidas

    def test_termina_sin_forzar(self):
        respuesta, modelo, recibidas = self.consultar([chat("<respuesta>2</respuesta>", "stop", "pienso")])
        self.assertEqual(respuesta, "<respuesta>2</respuesta>")
        self.assertFalse(modelo.ultimo_respuesta_forzada)
        self.assertEqual(len(recibidas), 1)
        self.assertEqual(recibidas[0][1]["max_tokens"], 100)

    def test_corte_fuerza_la_respuesta(self):
        respuesta, modelo, recibidas = self.consultar([chat("", "length", "pienso"), completado("1 3", "stop")])
        self.assertEqual(respuesta, "<respuesta>1 3</respuesta>")
        self.assertEqual(evaluar_respuesta(CAMINO, respuesta).recompensa, 1.0)
        self.assertTrue(modelo.ultimo_respuesta_forzada)
        self.assertEqual(modelo.ultimo_fin, "length")
        self.assertEqual(modelo.ultimo_tokens, 13)
        ruta, cuerpo = recibidas[1]
        self.assertEqual(ruta, "/v1/completions")
        self.assertEqual(cuerpo["max_tokens"], 7)
        self.assertEqual(cuerpo["stop"], ["<"])
        self.assertEqual(
            cuerpo["prompt"],
            "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n<think>\npienso\n</think>\n\n<respuesta>",
        )

    def test_razonamiento_en_el_contenido_visible(self):
        _, _, recibidas = self.consultar([chat("<think>\npienso", "length"), completado("1", "stop")])
        self.assertEqual(recibidas[1][1]["prompt"].count("<think>"), 1)
        self.assertTrue(recibidas[1][1]["prompt"].endswith("pienso\n</think>\n\n<respuesta>"))

        _, _, recibidas = self.consultar([chat("sigo", "length", "pienso\n</think>"), completado("1", "stop")])
        self.assertEqual(recibidas[1][1]["prompt"].count("</think>"), 1)

    def test_respuesta_cortada_no_se_cierra(self):
        respuesta, _, _ = self.consultar([chat("", "length", "pienso"), completado("1 2 3", "length")])
        self.assertEqual(evaluar_respuesta(CAMINO, respuesta).recompensa, -1.0)

    def test_bloque_completo_no_se_fuerza(self):
        respuesta, modelo, recibidas = self.consultar([chat("<respuesta>2</respuesta> y mas", "length", "pienso")])
        self.assertEqual(evaluar_respuesta(CAMINO, respuesta).puntaje, 6)
        self.assertFalse(modelo.ultimo_respuesta_forzada)
        self.assertEqual(len(recibidas), 1)

    def test_bloque_dentro_del_razonamiento_sin_cerrar_se_fuerza(self):
        respuesta, modelo, _ = self.consultar([
            chat("<think>\nquizas <respuesta>2</respuesta>", "length"),
            completado("1 3", "stop"),
        ])
        self.assertEqual(respuesta, "<respuesta>1 3</respuesta>")
        self.assertTrue(modelo.ultimo_respuesta_forzada)


class PruebasFormatosLMStudio(unittest.TestCase):
    def consultar(self, formato, respuestas):
        with ServidorSimulado(respuestas) as servidor:
            modelo = ModeloLMStudio(url_base=servidor.url, formato=formato)
            return modelo.generar("P"), servidor.recibidas

    def prompt_forzado(self, formato, contenido, razonamiento=None):
        _, recibidas = self.consultar(formato, [chat(contenido, "length", razonamiento), completado("1", "stop")])
        return recibidas[1][1]["prompt"]

    def test_modelos_sin_canal_de_razonamiento(self):
        self.assertEqual(
            self.prompt_forzado(OLMO, "pienso"),
            "<|user|>\nP\n<|assistant|>\npienso\n<respuesta>",
        )
        prompt = self.prompt_forzado(LLAMA, "pienso")
        self.assertTrue(prompt.endswith(
            "user<|end_header_id|>\n\nP<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\npienso\n<respuesta>"
        ))

    def test_deepseek_abre_el_razonamiento_en_la_plantilla(self):
        self.assertEqual(
            self.prompt_forzado(DEEPSEEK, "", "pienso"),
            "<｜User｜>P<｜Assistant｜><think>\npienso\n</think>\n\n<respuesta>",
        )

    def test_ministral_repone_su_etiqueta_y_la_cierra(self):
        for contenido, razonamiento in [("", "pienso"), ("[THINK]pienso", None)]:
            prompt = self.prompt_forzado(MINISTRAL, contenido, razonamiento)
            self.assertTrue(prompt.endswith("[/INST][THINK]pienso[/THINK]<respuesta>"))
            self.assertNotIn("<think>", prompt)

    def test_ministral_no_toma_el_bloque_del_razonamiento(self):
        respuesta, recibidas = self.consultar(MINISTRAL, [
            chat("[THINK]quizas <respuesta>2</respuesta>", "length"),
            completado("1 3", "stop"),
        ])
        self.assertEqual(respuesta, "<respuesta>1 3</respuesta>")
        self.assertEqual(len(recibidas), 2)

    def test_ministral_devuelve_lo_que_sigue_al_cierre(self):
        respuesta, _ = self.consultar(MINISTRAL, [chat("[THINK]dudo[/THINK]<respuesta>2</respuesta>", "stop")])
        self.assertEqual(respuesta, "<respuesta>2</respuesta>")


@unittest.skipUnless(find_spec("torch"), "Requiere torch")
class PruebasPresupuestoTransformers(unittest.TestCase):
    def consultar(self, pensamiento, respuesta, tokens_pensamiento=20, tokens_respuesta=8):
        import torch

        prompt = "P<think>\n"

        class Entrada(dict):
            def to(self, _):
                return self

        class Tokenizador:
            def apply_chat_template(self, *argumentos, tokenize=True, **opciones):
                return self(prompt) if tokenize else prompt

            def __call__(self, texto, **opciones):
                return Entrada(input_ids=torch.tensor([[ord(c) for c in texto]]))

            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(i) for i in ids.tolist() if i != 0)

        class Modelo:
            # Cada carácter es un token, y el 0 marca el fin del texto.
            device = "cpu"

            def __init__(self):
                self.llamadas = []

            def generate(self, input_ids, max_new_tokens, **opciones):
                self.llamadas.append(max_new_tokens)
                generado = "".join(chr(i) for i in input_ids[0].tolist())[len(prompt):]
                if generado.endswith("<respuesta>"):
                    nuevo = respuesta
                else:
                    nuevo = pensamiento[len(generado):]
                ids = [ord(c) for c in nuevo[:max_new_tokens]]
                if len(ids) < max_new_tokens:
                    ids.append(0)
                return torch.cat([input_ids, torch.tensor([ids], dtype=torch.long)], dim=1)

        modelo = ModeloTransformers("x", razonar=True, tokens_pensamiento=tokens_pensamiento, tokens_respuesta=tokens_respuesta)
        modelo._modelo, modelo._tokenizador = Modelo(), Tokenizador()
        return modelo.generar("hola"), modelo

    def test_termina_sin_forzar(self):
        texto, modelo = self.consultar("a</think><respuesta>2</respuesta>", "", tokens_pensamiento=60)
        self.assertEqual(texto, "<respuesta>2</respuesta>")
        self.assertFalse(modelo.ultimo_respuesta_forzada)
        self.assertEqual(modelo._modelo.llamadas, [60])

    def test_corte_fuerza_la_respuesta(self):
        texto, modelo = self.consultar("p" * 50, "1 3</respuesta>", tokens_respuesta=30)
        self.assertEqual(texto, "<respuesta>1 3</respuesta>")
        self.assertTrue(modelo.ultimo_respuesta_forzada)
        self.assertEqual(modelo.ultimo_fin, "length")
        self.assertEqual(modelo._modelo.llamadas, [20, 30])
        self.assertIn("</think>", modelo.ultimo_texto)

    def test_fin_sin_cierre_se_completa(self):
        texto, _ = self.consultar("p" * 50, "1 3", tokens_respuesta=30)
        self.assertEqual(texto, "<respuesta>1 3</respuesta>")

    def test_respuesta_cortada_no_se_cierra(self):
        texto, _ = self.consultar("p" * 50, "1 2 3 4 5 6 7 8", tokens_respuesta=5)
        self.assertEqual(evaluar_respuesta(CAMINO, texto).recompensa, -1.0)


class PruebasEvaluarDataset(unittest.TestCase):
    def test_usa_los_arboles_del_archivo_y_reanuda(self):
        ejemplos = generar_datasets(semilla=2, entrenamiento_por_nivel=1, evaluacion_por_nivel=1, test_por_nivel=1)["test"]
        with tempfile.TemporaryDirectory() as carpeta:
            datos = Path(carpeta) / "test.jsonl"
            ruta = Path(carpeta) / "sub" / "r.jsonl"
            guardar_datasets({"test": ejemplos}, carpeta)

            modelo = ModeloFijo(falla_en=2)
            evaluar_dataset(modelo, "a", datos, ruta)
            self.assertEqual(len(leer_resultados(ruta)), len(ejemplos) - 1)  # La falla no se guarda
            self.assertEqual(sorted(modelo.prompts), sorted(e["prompt"] for e in ejemplos))

            evaluar_dataset(modelo, "a", datos, ruta)
            self.assertEqual(len(leer_resultados(ruta)), len(ejemplos))
            self.assertEqual(len(modelo.prompts), len(ejemplos) + 1)  # Solo reintenta la pendiente

            resumen = resumir(ruta)
            self.assertEqual(len(resumen), len(NIVELES))
            self.assertTrue(((resumen["episodios"] == 1) & (resumen["recompensa"] == 0.0)).all())

    def test_evalua_varios_modelos_y_reintenta_las_fallas(self):
        class ModeloRoto(ModeloLLM):
            def generar(self, prompt):
                raise ValueError("error inesperado")

        ejemplos = generar_datasets(semilla=2, entrenamiento_por_nivel=1, evaluacion_por_nivel=1, test_por_nivel=1)["test"]
        with tempfile.TemporaryDirectory() as carpeta:
            guardar_datasets({"test": ejemplos}, carpeta)
            datos = Path(carpeta) / "test.jsonl"
            rutas = [Path(carpeta) / nombre / "r.jsonl" for nombre in "abc"]
            evaluaciones = [
                (ModeloFijo(falla_en=2), "a", rutas[0]),
                (ModeloRoto(), "b", rutas[1]),
                (ModeloFijo(), "c", rutas[2]),
            ]

            pendientes = evaluar_varios(evaluaciones, datos, rondas=2, segundos_pausa=0)

            self.assertEqual(pendientes, ["b"])
            self.assertEqual(len(leer_resultados(rutas[0])), len(ejemplos))  # La falla se recuperó en la segunda ronda
            self.assertEqual(leer_resultados(rutas[1]), [])
            self.assertEqual(len(leer_resultados(rutas[2])), len(ejemplos))
            self.assertEqual(len(evaluaciones[2][0].prompts), len(ejemplos))  # Un modelo completo no se repite
            self.assertEqual(set(resumir(rutas).index.get_level_values("modelo")), {"a", "c"})
            self.assertEqual(set(resumir_pass(rutas, k=1).index.get_level_values("modelo")), {"a", "c"})

    def test_resumen_cuenta_cortes_e_invalidos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "r.jsonl"
            base = dict(modelo="a", nivel=1, semilla=None, puntaje_voraz=4, puntaje_optimo=10, puntaje_sin_vecinos=10, segundos=2.0, tokens=50)
            filas = [
                dict(base, episodio=1, recompensa=1.0, puntaje=10, fin="stop"),
                dict(base, episodio=2, recompensa=-1.0, puntaje=None, fin="length"),
            ]
            ruta.write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
            fila = resumir(ruta).loc[("a", 1)]
            self.assertEqual((fila["recompensa"], fila["optimos"], fila["supera_voraz"]), (0.0, 0.5, 0.5))
            self.assertEqual((fila["invalidos"], fila["cortes"], fila["optimos_atajo"]), (0.5, 0.5, 1.0))
            self.assertEqual(resumir(ruta, por=["modelo"]).loc["a", "episodios"], 2)

    def test_resumen_acepta_resultados_anteriores(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "r.jsonl"
            fila = dict(modelo="a", nivel=1, recompensa=1.0, puntaje=10, puntaje_voraz=4, puntaje_optimo=10, segundos=2.0, tokens=50)
            ruta.write_text(json.dumps(fila) + "\n", encoding="utf-8")
            resumen = resumir(ruta).loc[("a", 1)]
            self.assertEqual((resumen["optimos"], resumen["cortes"]), (1.0, 0.0))
            self.assertTrue(resumen.isna()["optimos_atajo"])

    def test_intentos_niveles_y_pass_at_k(self):
        ejemplos = generar_datasets(semilla=3, entrenamiento_por_nivel=1, evaluacion_por_nivel=1, test_por_nivel=2)["test"]
        with tempfile.TemporaryDirectory() as carpeta:
            datos = Path(carpeta) / "test.jsonl"
            ruta = Path(carpeta) / "r.jsonl"
            guardar_datasets({"test": ejemplos}, carpeta)

            modelo = ModeloFijo()
            evaluar_dataset(modelo, "a", datos, ruta, niveles=[2, 3], intentos=3)
            self.assertEqual(len(leer_resultados(ruta)), 2 * 2 * 3)
            evaluar_dataset(modelo, "a", datos, ruta, niveles=[2, 3], intentos=3)
            self.assertEqual(len(modelo.prompts), 2 * 2 * 3)

    def test_calculo_de_pass_at_k(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "r.jsonl"
            # Árbol A acierta en el intento 2, B nunca, y C tiene solo un intento (acertado).
            intentos = [("A", 1, 0.5), ("A", 2, 1.0), ("A", 3, 0.5), ("B", 1, 0.0), ("B", 2, -1.0), ("B", 3, 0.2), ("C", 1, 1.0)]
            filas = [dict(modelo="m", nivel=2, episodio=e, intento=i, recompensa=r) for e, i, r in intentos]
            ruta.write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
            fila = resumir_pass(ruta, k=3).loc[("m", 2)]
            self.assertEqual(fila["intentos"], 7)
            self.assertAlmostEqual(fila["pass@1"], round(2 / 7, 3))
            self.assertEqual(fila["arboles_completos"], 2)
            self.assertEqual(fila["pass@3"], 0.5)
            self.assertEqual(resumir_pass(ruta, k=3, por=["modelo"]).loc["m", "pass@3"], 0.5)

    def test_pass_at_k_usa_todos_los_intentos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "r.jsonl"
            # D tiene 6 intentos y solo acierta en el quinto: no basta con mirar los 3 primeros.
            filas = [dict(modelo="m", nivel=2, episodio="D", intento=i, recompensa=1.0 if i == 5 else 0.0) for i in range(1, 7)]
            ruta.write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
            fila = resumir_pass(ruta, k=3).loc[("m", 2)]
            self.assertAlmostEqual(fila["pass@3"], 0.5)  # 1 - C(5, 3) / C(6, 3)
            self.assertAlmostEqual(resumir_pass(ruta, k=4).loc[("m", 2), "pass@4"], round(1 - 5 / 15, 3))
            self.assertEqual(resumir_pass(ruta, k=6).loc[("m", 2), "pass@6"], 1.0)
            self.assertTrue(resumir_pass(ruta, k=7).isna().loc[("m", 2), "pass@7"])

    def test_resumen_con_los_primeros_intentos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "r.jsonl"
            # Igual que arriba: con solo los 4 primeros intentos, D nunca acierta.
            filas = [
                dict(modelo="m", nivel=2, episodio="D", intento=i, recompensa=1.0 if i == 5 else 0.0,
                     puntaje=0, puntaje_voraz=0, puntaje_optimo=1, segundos=1.0, tokens=1)
                for i in range(1, 7)
            ]
            ruta.write_text("".join(json.dumps(f) + "\n" for f in filas), encoding="utf-8")
            fila = resumir_pass(ruta, k=4, intentos=4).loc[("m", 2)]
            self.assertEqual(fila["intentos"], 4)
            self.assertEqual(fila["pass@4"], 0.0)
            self.assertEqual(resumir(ruta, intentos=4).loc[("m", 2), "episodios"], 4)
            self.assertEqual(resumir(ruta).loc[("m", 2), "episodios"], 6)

    def test_sin_vecinos(self):
        self.assertEqual(CAMINO.puntaje_sin_vecinos(), 6)  # Toma el 6 y sus vecinos quedan fuera
        self.assertEqual(Instancia((-1, -2), ((1, 2),), 1).puntaje_sin_vecinos(), 0)


class PruebasDataset(unittest.TestCase):
    def test_agrega_un_arbol_nuevo_por_nivel(self):
        datasets = generar_datasets(semilla=4, entrenamiento_por_nivel=2, evaluacion_por_nivel=1, test_por_nivel=1)
        with tempfile.TemporaryDirectory() as carpeta:
            guardar_datasets(datasets, carpeta)
            ruta = Path(carpeta) / "extra.jsonl"
            primeros = agregar_arboles(ruta, carpeta)
            # Con otra forma de escribir la misma ruta, el archivo no se lee dos veces.
            segundos = agregar_arboles(Path(os.path.relpath(ruta)), carpeta)

            self.assertEqual([e["nivel"] for e in primeros], list(range(1, len(NIVELES) + 1)))
            self.assertEqual(primeros[0]["id"], "extra-1-0")
            self.assertEqual(segundos[0]["id"], "extra-1-1")
            self.assertEqual(leer_resultados(ruta), primeros + segundos)
            todos = [e for ejemplos in datasets.values() for e in ejemplos] + primeros + segundos
            formas = {forma_canonica(Instancia(tuple(e["valores"]), tuple(map(tuple, e["aristas"])), e["costo"])) for e in todos}
            self.assertEqual(len(formas), len(todos))

    def test_resolver_coincide_con_verificador(self):
        rng = random.Random(7)
        for _ in range(1500):
            nivel = Nivel(rng.randint(1, 45), -30, 40, 12, voraz_maximo=1.0)
            instancia = generar_instancia(nivel, rng)
            seleccion, razonamiento = resolver(instancia)
            self.assertEqual(instancia.puntaje(seleccion), instancia.puntaje_optimo(), instancia)
            self.assertIn(f"The best score is {instancia.puntaje_optimo()}.", razonamiento)

    def test_forma_canonica_ignora_la_numeracion(self):
        original = Instancia((5, 6, 5), ((1, 2), (2, 3)), 2)
        renumerado = Instancia((6, 5, 5), ((3, 1), (1, 2)), 2)
        distinto = Instancia((5, 5, 6), ((1, 2), (2, 3)), 2)
        self.assertEqual(forma_canonica(original), forma_canonica(renumerado))
        self.assertNotEqual(forma_canonica(original), forma_canonica(distinto))

    def test_partes_correctas_y_sin_arboles_en_comun(self):
        tamanos = dict(entrenamiento_por_nivel=10, evaluacion_por_nivel=3, test_por_nivel=4)
        datasets = generar_datasets(semilla=1, **tamanos)
        self.assertEqual(len(datasets["evaluacion"]), 3 * len(NIVELES))
        self.assertEqual(len(datasets["test"]), 4 * len(NIVELES))
        vistos = set()
        for ejemplos in datasets.values():
            self.assertEqual({e["nivel"] for e in ejemplos}, set(range(1, len(NIVELES) + 1)))
            for e in ejemplos:
                instancia = Instancia(tuple(e["valores"]), tuple(map(tuple, e["aristas"])), e["costo"])
                forma = forma_canonica(instancia)
                self.assertNotIn(forma, vistos)
                vistos.add(forma)
                self.assertEqual(e["prompt"], generar_prompt(instancia))
                self.assertEqual(evaluar_respuesta(instancia, e["respuesta"]).recompensa, 1.0)
        self.assertEqual(generar_datasets(semilla=1, **tamanos), datasets)


if __name__ == "__main__":
    unittest.main()
