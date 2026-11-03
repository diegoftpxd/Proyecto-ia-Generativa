
from abc import ABC, abstractmethod
from typing import Self

class Pieza(ABC):
    letra = "None"
    def __init__(self, x:int, y:int, color:str="Blanco", casillas_objetivo=None):
        self.x:int = x
        self.y:int = y
        if casillas_objetivo == None:
            self.casillas_objetivo:set = self.obtener_posiciones_donde_puedo_llegar()
        else:
            self.casillas_objetivo = casillas_objetivo
        self.color = color

    @property
    def posicion(self) -> tuple[int, int]:
        return self.x, self.y
    

    def se_puede_mover_a(self, posicion_objetivo) -> bool:
        return posicion_objetivo in self.casillas_objetivo
    
    def mover(self, posicion_objetivo):
        self.x = posicion_objetivo[0]
        self.y = posicion_objetivo[1]
        self.casillas_objetivo = self.obtener_posiciones_donde_puedo_llegar()

    def copy(self) -> Self:
        return self.__class__(self.x, self.y, self.color, self.casillas_objetivo.copy())


    @abstractmethod
    def obtener_posiciones_donde_puedo_llegar(self) -> set:
        pass

    def __repr__(self) -> set:
        return self.__class__.letra + self.color[0]
        



class Torre(Pieza):
    letra = "T"
    def obtener_posiciones_donde_puedo_llegar(self) -> set:
        return {
            (self.x + 1, self.y),
            (self.x, self.y + 1),
            (self.x - 1, self.y),
            (self.x, self.y - 1) 
        }


class Arfil(Pieza):
    letra = "A"
    def obtener_posiciones_donde_puedo_llegar(self) -> set:
        return {
            (self.x + 1, self.y + 1),
            (self.x - 1, self.y + 1),
            (self.x - 1, self.y - 1),
            (self.x + 1, self.y - 1) 
        }


class Caballo(Pieza):
    letra = "C"
    def obtener_posiciones_donde_puedo_llegar(self) -> set:
        return {
            (self.x + 2, self.y + 1),
            (self.x + 2, self.y - 1),
            (self.x - 2, self.y + 1),
            (self.x - 2, self.y - 1),
            (self.x + 1, self.y + 2),
            (self.x + 1, self.y - 2),
            (self.x - 1, self.y + 2),
            (self.x - 1, self.y - 2),
        }