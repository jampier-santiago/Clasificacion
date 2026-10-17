# Clasificación: comparación de algoritmos de Machine Learning

**Semana 8 · CADI · Introducción a Machine Learning** · Universidad de Cundinamarca
**Autor:** Jampier Santiago Moreno Arenas

Este proyecto aplica cinco algoritmos de clasificación al mismo problema, con la misma partición de datos y las mismas métricas. El objetivo es compararlos de forma justa y decidir cuál usar.

> **Alcance:** el dataset se usa solo con fines académicos. Los resultados no sirven como herramienta clínica.

## Problema y datos

- **Dataset:** Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`). No hay que descargar nada.
- **Tamaño:** 569 observaciones y 30 variables numéricas continuas. Son la media, el error estándar y el valor "worst" de 10 medidas del núcleo celular, como radio, textura, perímetro, área y concavidad.
- **Variable objetivo:** en sklearn `0` es maligno. Aquí se invierte para que la **clase positiva (1) sea maligno**. Hay 357 casos benignos (62,7 %) y 212 malignos (37,3 %).
- **Partición:** 80 % train y 20 % test, estratificada, con `random_state=42`. El test tiene 114 observaciones, 42 de ellas malignas.

## Modelos evaluados

| Modelo | Configuración |
|---|---|
| Decision Tree | `max_depth=4` |
| Random Forest | `n_estimators=100` |
| SVM | kernel RBF, `C=1.0`, `gamma='scale'`, con `StandardScaler` en un Pipeline |
| K-NN | `n_neighbors=5`, con `StandardScaler` en un Pipeline |
| Naive Bayes | `GaussianNB`, sin escalamiento |

Como línea base opcional se incluye una Regresión Logística. Todos los modelos usan la misma función de evaluación, que calcula Accuracy, Precision, Recall, F1 y la matriz de confusión.

## Resultados (conjunto de prueba)

| Modelo | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| Decision Tree | 0,9123 | 0,9444 | 0,8095 | 0,8718 | 2 | 8 |
| Random Forest | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0 | 3 |
| SVM | 0,9737 | 1,0000 | 0,9286 | 0,9630 | 0 | 3 |
| K-NN | 0,9561 | 0,9744 | 0,9048 | 0,9383 | 1 | 4 |
| Naive Bayes | 0,9386 | 1,0000 | 0,8333 | 0,9091 | 0 | 7 |

### Modelo final: SVM (kernel RBF con `StandardScaler`)

SVM y Random Forest empatan en el test. Para desempatar sin tocar el conjunto de prueba se usó validación cruzada estratificada de 5 pliegues solo sobre el train. SVM quedó levemente arriba en Accuracy, Recall y F1, y con menos variación entre pliegues. En un problema diagnóstico el falso negativo es el error más costoso, así que se prioriza el Recall sin sacrificar la Precision.

Aun así quedan 3 malignos sin detectar. Con tan pocos casos de prueba la ventaja de SVM sobre Random Forest es pequeña, y Random Forest sería una alternativa igual de defendible. El análisis completo está en las secciones 7 y 8 del notebook.

## Estructura del proyecto

```
Clasificacion/
├── Semana8_Clasificacion_ML.ipynb   # Notebook con el código, el análisis y las conclusiones
├── README.md
├── matrices_confusion.png           # Se genera al ejecutar el notebook
└── comparacion_metricas.png         # Se genera al ejecutar el notebook
```

## Requisitos

- Python 3.9 o superior (probado con 3.12)
- numpy, pandas, matplotlib, scikit-learn
- ipykernel / jupyter para ejecutar el notebook

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn ipykernel jupyter
```

## Cómo ejecutarlo

**En VS Code:** abre el notebook, elige el kernel `.venv` (Select Kernel → Python Environments) y ejecuta con **Run All**.

**En Jupyter:**

```bash
jupyter notebook Semana8_Clasificacion_ML.ipynb
```

Si una celda falla con `ModuleNotFoundError`, casi siempre es porque el kernel seleccionado no es el del entorno donde se instalaron los paquetes.

## Contenido del notebook

1. Preparación del caso y los datos (clase positiva, partición estratificada, función de evaluación)
2. Decision Tree
3. Random Forest
4. SVM
5. K-NN
6. Naive Bayes
7. Comparación y toma de decisión (tablas, gráficas y preguntas de análisis)
8. Modelo final y justificación (validación cruzada, línea base y conclusiones)

## Reproducibilidad

Todas las semillas están fijadas con `random_state=42`, así que los resultados deberían coincidir entre ejecuciones. Pueden variar levemente con otras versiones de scikit-learn.
