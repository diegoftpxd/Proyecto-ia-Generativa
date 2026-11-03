from dataclasses import dataclass

from verificador import calcular_optimo


@dataclass(frozen=True)
class Instancia:
    # Los nodos se numeran desde 1: valores[0] es el valor del nodo 1.
    valores: tuple[int, ...]
    aristas: tuple[tuple[int, int], ...]
    costo: int

    @property
    def cantidad_nodos(self) -> int:
        return len(self.valores)

    def vecinos(self) -> dict[int, list[int]]:
        vecinos = {nodo: [] for nodo in range(1, self.cantidad_nodos + 1)}
        for a, b in self.aristas:
            vecinos[a].append(b)
            vecinos[b].append(a)
        return vecinos

    def puntaje(self, seleccion: set[int]) -> int:
        # Cada nodo elegido aporta su valor, menos el costo por cada vecino también elegido.
        vecinos = self.vecinos()
        return sum(
            self.valores[nodo - 1] - self.costo * sum(vecino in seleccion for vecino in vecinos[nodo])
            for nodo in seleccion
        )

    def puntaje_voraz(self) -> int:
        # La regla simple: elegir todos los nodos positivos, o ninguno si eso resta puntos.
        positivos = {i + 1 for i, valor in enumerate(self.valores) if valor > 0}
        return max(0, self.puntaje(positivos))

    def puntaje_optimo(self) -> int:
        # El verificador numera los nodos desde 0.
        aristas = [[] for _ in self.valores]
        for a, b in self.aristas:
            aristas[a - 1].append(b - 1)
            aristas[b - 1].append(a - 1)
        return calcular_optimo(self.cantidad_nodos, self.costo, list(self.valores), aristas)



