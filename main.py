import numpy as np
import matplotlib.pyplot as plt

def GenerateSimplePlot(mass_array, frequency_array, img_name="mass_freq.png"):  # Generar la imagen sin el ajuste
    plt.plot(mass_array, frequency_array)

    plt.plot(mass_array, frequency_array, 'ro')   # Crear el grafico, en el eje x la masa y en el eje y la frecuencia
    plt.xlabel('masa')
    plt.ylabel('frecuencia')
    plt.title('Frecuencia VS Masa')
    plt.savefig("images/" + img_name)
    plt.close()
def GenerateLinearRegressionPlot(mass_array, frequency_array, img_name="linear_regression.png"):
    # Ajuste de regresión lineal
    coefficients = np.polyfit(mass_array, frequency_array, 1)
    m, b = coefficients

    # Calcular los valores de y para la línea de regresión
    line_fit = m * np.array(mass_array) + b

    # Crear el gráfico con el ajuste de regresión lineal
    plt.plot(mass_array, frequency_array, 'ro', label='Datos')
    plt.plot(mass_array, line_fit, label='Regresión lineal')
    plt.xlabel('masa')
    plt.ylabel('frecuencia')
    plt.title('Frecuencia VS Masa - ajuste de regresión lineal')
    plt.legend()
    plt.savefig("images/" + img_name)


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
    data_matrix = ReadCsv("calculo.csv")
    x = data_matrix[:, 0]
    y = data_matrix[:, 1]

    GenerateSimplePlot(x, y, "mass_freq_random.png")
    GenerateLinearRegressionPlot(x,y,"linear_regression.png")
