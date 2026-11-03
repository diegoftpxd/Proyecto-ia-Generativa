from instancia import Instancia


def generar_prompt(instancia: Instancia) -> str:
    valores = "\n".join(f"{i}: {valor}" for i, valor in enumerate(instancia.valores, 1))
    aristas = "\n".join(f"{a}-{b}" for a, b in instancia.aristas) or "(ninguna)"



    return f"""Elige un conjunto de nodos de este árbol para obtener el mayor puntaje posible.

Valor de cada nodo:
{valores}

Conexiones:
{aristas}

Costo c = {instancia.costo}

Puntaje: cada nodo elegido aporta su valor menos c por cada vecino que también esté elegido. Los nodos no elegidos no aportan nada. Si no eliges ninguno, el puntaje es 0.

Termina con los números de los nodos elegidos, por ejemplo:
<respuesta>2 5 9</respuesta>
Si no eliges ninguno, escribe <respuesta></respuesta>."""
