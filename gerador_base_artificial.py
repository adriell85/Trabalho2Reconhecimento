import numpy as np
import pandas as pd

def gerar_dataset():
    # Parâmetros para as amostras da classe 0
    int_range_classe_0 = (0, 50)
    float_range_classe_0 = (0.0, 5.0)

    # Parâmetros para as amostras da classe 1
    int_range_classe_1 = (51, 100)
    float_range_classe_1 = (5.1, 10.0)

    # Gerar amostras para a classe 0
    int_classe_0 = np.random.randint(*int_range_classe_0, 30)
    float_classe_0 = np.random.uniform(*float_range_classe_0, 30)
    labels_classe_0 = np.zeros(30, dtype=int)

    # Gerar amostras para a classe 1
    int_classe_1 = np.random.randint(*int_range_classe_1, 10)
    float_classe_1 = np.random.uniform(*float_range_classe_1, 10)
    labels_classe_1 = np.ones(10, dtype=int)

    # Combinar as amostras das duas classes
    atributos_int = np.concatenate((int_classe_0, int_classe_1))
    atributos_float = np.concatenate((float_classe_0, float_classe_1))
    labels = np.concatenate((labels_classe_0, labels_classe_1))

    # Criar o DataFrame
    dataset = pd.DataFrame({
        'Atributo_Inteiro': atributos_int,
        'Atributo_Float': atributos_float,
        'Classe': labels
    })

    return dataset

# Gerar o dataset
dataset_exemplo = gerar_dataset()
dataset_exemplo.head()  # Mostrar as primeiras linh

print(dataset_exemplo)