# Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Definimos funciones
def GeneratePlotsinAjuste(mass_array, frequency_array, img_name="frecuencia_vs_masa.png"):  # Generar la imagen sin el ajuste
    plt.plot(mass_array, frequency_array)
    plt.plot(mass_array, frequency_array, 'ro')   # Crear el grafico, en el eje "x": la masa y en el eje "y": la frecuencia
    plt.xlabel('Masas (Kg)')
    plt.ylabel('Frecuencias (Hz)')
    plt.title('Frecuencia VS Masa')
    plt.savefig(img_name)               # Guarda la imagen de la grafica
    plt.close()

def GenerateLinearRegressionPlot(mass_array, frequency_array, img_name="frecuencia_vs_masa_linear_regression.png"):
    # Ajuste de regresión lineal
    coefficients = np.polyfit(mass_array, frequency_array, 1)       # con polyfit genero un ajuste de regresion lineal (1) a mis datos
    m, b = coefficients
    # Calcular los valores de "y" para la línea de regresión
    line_fit = m * np.array(mass_array) + b                         # Genero la funcion lineal que me botara las frecuencias ajustadas linealmente
    # Crear el gráfico con el ajuste de regresión lineal
    plt.plot(mass_array, frequency_array, 'ro')       # Grafico la masa y frecuencia original
    plt.plot(mass_array, line_fit, label='Ajuste de regresión lineal')         # Grafico la masa, y la frecuencia ajustada a una regresion lineak
    plt.xlabel('Masas (Kg)')
    plt.ylabel('Frecuencias (Hz)')
    plt.title('Frecuencia VS Masa - ajuste de regresión lineal')
    plt.legend()
    plt.savefig(img_name)
    print("Ajuste Regresion Lineal:")
    print("y = mx + b")
    print("y = " + str(m) + "x + " + str(b))
    print("Parametros de ajuste:")
    print("m = " + str(m))
    print("b = " + str(b))
    r2 = r2_score(frequency_array, line_fit)
    print(f"Coeficiente de determinación (R²): {r2}\n")
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
    plt.xlabel('Masas (Kg)')
    plt.ylabel('Frecuencias (Hz)')
    plt.title('Ajuste de regresión cuadrática')
    plt.legend()
    plt.savefig(img_name)
    r2 = r2_score(frequency_array,  p(mass_array))
    print("Ajuste Regresion Cuadratica:")
    print("y = ax^2 + bx + c")
    print("y = " + str(coef[0]) + "x^2 - " + str(-1*coef[1]) + "x + " + str(coef[2]))
    print("Parametros de ajuste:")
    print("a = " +str(coef[0]))
    print("b = " +str(coef[1]))
    print("c = " +str(coef[2]))
    print(f"Coeficiente de determinación (R²): {r2}\n")
    plt.close()

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c
def GenerateRegressionNoLineal(mass_array, frequency_array, img_name="exponencial_regresion_exponencial.png"):
    popt, _ = curve_fit(exponential_func, mass_array, frequency_array)
    # Valores para la curva ajustada
    frecuencias_fit = exponential_func(mass_array, *popt)
    # Gráfico de los datos y la curva ajustada
    plt.plot(mass_array, frecuencias_fit, label='Ajuste de regresión no lineal')
    plt.plot(mass_array, frequency_array, 'ro')
    plt.xlabel('Masas (Kg)')
    plt.ylabel('Frecuencias (Hz)')
    plt.title('Ajuste de regresión no lineal - Función exponencial')
    plt.legend()
    plt.savefig(img_name)
    r2 = r2_score(frequency_array,  frecuencias_fit)
    print("Ajuste Regresion No Lineal - Exponencial:")
    print("y = a * e^(bx) + c")
    print("y = " + str(popt[0]) + " * e^(" + str(popt[1]) + "x) + " + str(popt[2]))
    print("Parametros de ajuste:")
    print("a = " +str(popt[0]))
    print("b = " +str(popt[1]))
    print("c = " +str(popt[2]))
    print(f"Coeficiente de determinación (R²): {r2}\n")
    plt.close()

if __name__ == '__main__':  # Inicializar el programa
    data_matrix = np.loadtxt('datos.txt', delimiter=',')
    x = data_matrix[:, 0]       # masas de la data (g.)
    x_kg = x/1000               # por conveniencia y para que al modelar la exponencial no haya desbordamiento de datos lo pasaremos todo a kg,
                                #  aunque en gramos es lo mismo ya que tienen el mismo R² y misma grafica
    y = data_matrix[:, 1]       # frecuencias de la data (Hz)
    import pylatexenc

    GeneratePlotsinAjuste(x_kg, y, "frecuencia_vs_masa.png")
    GenerateLinearRegressionPlot(x_kg,y,"linear_regression.png")
    GenerateCuadraticaRegressionPlot(x_kg,y,"cuadratica_regresion.png")
    GenerateRegressionNoLineal(x_kg,y,"exponencial_regresion_exponencial.png")
