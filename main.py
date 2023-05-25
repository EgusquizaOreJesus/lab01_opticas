
# Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score

# Definimos funciones
def GeneratePlotsinAjuste(mass_array, frequency_array, img_name="frecuencia_vs_masa.png"):  # Generar la imagen sin el ajuste
    plt.plot(mass_array, frequency_array)
    plt.plot(mass_array, frequency_array, 'ro')   # Crear el grafico, en el eje "x": la masa y en el eje "y": la frecuencia
    plt.xlabel('masa')
    plt.ylabel('frecuencia')
    plt.title('Frecuencia VS Masa')
    plt.savefig("images/" + img_name)               # Guarda la imagen de la grafica
    plt.close()

def GenerateLinearRegressionPlot(mass_array, frequency_array, img_name="frecuencia_vs_masa_linear_regression.png"):
    # Ajuste de regresión lineal
    coefficients = np.polyfit(mass_array, frequency_array, 1)       # con polyfit genero un ajuste de regresion lineal (1) a mis datos
    m, b = coefficients
    # Calcular los valores de "y" para la línea de regresión
    line_fit = m * np.array(mass_array) + b                         # Genero la funcion lineal que me botara las frecuencias ajustadas linealmente
    # Crear el gráfico con el ajuste de regresión lineal
    plt.plot(mass_array, frequency_array, 'ro', label='Datos')       # Grafico la masa y frecuencia original
    plt.plot(mass_array, line_fit, label='Regresión lineal')         # Grafico la masa, y la frecuencia ajustada a una regresion lineak
    plt.xlabel('masa')
    plt.ylabel('frecuencia')
    plt.title('Frecuencia VS Masa - ajuste de regresión lineal')
    plt.legend()
    plt.savefig("images/" + img_name)
    print("Ajuste Regresion Lineal:")
    print("y = mx + b")
    print("y = " + str(m) + "x + " + str(b))
    print("Parametros de ajuste:")
    print("m = " + str(m))
    print("b = " + str(b))
    r2 = r2_score(frequency_array, line_fit)
    print("COEFICIENTE DE DETERMINACION r^2:")
    print(f"r^2 = {r2}\n")
    plt.close()

def GenerateCuadraticaRegressionPlot(mass_array, frequency_array, img_name="cuadratica_regression.png"):
    # Ajuste de regresión lineal
    coef = np.polyfit(mass_array, frequency_array, 2)           # con polyfit genero un ajuste de regresion cuadratica (2) a mis datos
    p = np.poly1d(coef)                                         # con dicha funcion genero un polinomio a partir de los coeficientes
    # GENERACION DE DATOS PARA LA CURVA DEL AJUSTE
    masas_fit = np.linspace(min(mass_array), max(mass_array), 100)  # Genero un arreglo de valores espaciados uniformemente entre el valor minimo y maximo de masas
    frecuencias_fit = p(masas_fit)                                  # Calculo los valores de la frecuencia correspondientes a las masas brindadas en la linea anterior
                                                                    # Utilizando la funcion de regresion cuadratica "p"
    # Graficar el ajuste de regresión cuadrática
    plt.plot(masas_fit, frecuencias_fit, label='Ajuste de regresión cuadrática')        # Grafico la masa uniforme, y la frecuencia ajustada a una regresion cuadratica
    plt.plot(mass_array, frequency_array, 'ro')                                         # Grafico la masa y frecuencia original
    plt.xlabel('Masas')
    plt.ylabel('Frecuencias')
    plt.title('Ajuste de regresión cuadrática')
    plt.legend()
    plt.savefig("images/" + img_name)
    r2 = r2_score(frequency_array,  p(mass_array))
    print("Ajuste Regresion Cuadratica:")
    print("y = ax^2 + bx + c")
    print("y = " + str(coef[0]) + "x^2 - " + str(-1*coef[1]) + "x + " + str(coef[2]))
    print("Parametros de ajuste:")
    print("a = " +str(coef[0]))
    print("b = " +str(coef[1]))
    print("c = " +str(coef[2]))
    print("COEFICIENTE DE DETERMINACION r^2:")
    print(f"r^2 = {r2}")
    plt.close()

def ReadCsv(filename="calculo.csv"):  # Leer el archivo csv
    data_matrix = []

    import csv
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        for row in file:
            row[0] = int(row[0])
            row[1] = float(row[1])
            data_matrix.append(row)

    return np.array(data_matrix)  # Convertir el archivo a una matriz

if __name__ == '__main__':  # Inicializar el programa
    data_matrix = np.loadtxt('datos.txt', delimiter=',')
    x = data_matrix[:, 0]
    y = data_matrix[:, 1]

    GeneratePlotsinAjuste(x, y, "frecuencia_vs_masa.png")
    GenerateLinearRegressionPlot(x,y,"linear_regression.png")
    GenerateCuadraticaRegressionPlot(x,y,"cuadratica_regresion.png")
