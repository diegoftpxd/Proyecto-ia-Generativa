import random
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from entorno import EntornoSeleccion, calcular_recompensa, evaluar_respuesta
from evaluacion import evaluar_modelo, leer_resultados, resumir
from generador import NIVELES, Nivel, generar_instancia
from modelo_hf import extraer_respuesta
from instancia import Instancia
from modelo import ModeloLLM
from prompt import generar_prompt


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

    def test_reproducible(self):
        self.assertEqual(generar_instancia(NIVELES[4], random.Random(9)), generar_instancia(NIVELES[4], random.Random(9)))

    def test_parametros_invalidos(self):
        for argumentos in [(0, 1, 2, 1), (3, 5, 1, 1), (3, -5, 0, 1), (3, 1, 5, 0), (3, 1, 5, 1, 1.5), (3, 1, 5, 1, -0.1)]:
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
            (5, 0, 6, 0.5 + 0.5 * 5 / 6), (0, 0, 6, 0.0), (-2, 0, 6, 0.0),
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
        for texto in ["", "1 3", "<respuesta>1 3", "<respuesta>uno</respuesta>", "<respuesta>1 1</respuesta>",
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


if __name__ == "__main__":
    unittest.main()
