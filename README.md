# Proyecto Final — Deep Learning
## Analisis de Sentimientos en Resenas de Peliculas (IMDB)
### Maestria en Ciencia de Datos

---

## Descripcion General

Este repositorio contiene el notebook unificado y los resultados del proyecto final de Deep Learning, cuyo objetivo es construir un sistema progresivo de analisis de sentimientos sobre el dataset IMDB de 50,000 resenas de peliculas. El proyecto se estructura en cinco etapas que avanzan desde un modelo base hasta el despliegue de una aplicacion interactiva, demostrando la evolucion de las tecnicas de Deep Learning aplicadas a clasificacion binaria de texto.

La tarea central es predecir si una resena de pelicula expresa un sentimiento positivo o negativo.

---

## Dataset

- Nombre: IMDB Dataset of 50K Movie Reviews
- Fuente: HuggingFace Datasets / Kaggle (lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Tamano: 50,000 resenas (25,000 train / 25,000 test)
- Clases: Positivo (1) / Negativo (0)
- Distribucion: Balanceada (50% cada clase)
- Variables: texto de la resena y etiqueta de sentimiento

Para ejecutar en Kaggle, se debe agregar el dataset como Input desde el panel lateral buscando "imdb-dataset-of-50k-movie-reviews". Si no se encuentra, el notebook descarga automaticamente desde HuggingFace como alternativa.

---

## Estructura del Proyecto

El notebook esta organizado en cinco etapas progresivas, cada una construyendo sobre los resultados de la anterior.

### Etapa 1 — Modelo Base: MLP con TF-IDF

Se establece la linea base del proyecto utilizando un Multi-Layer Perceptron (MLP) alimentado con representaciones TF-IDF del texto.

Arquitectura del MLP:
- Entrada: 10,000 features TF-IDF (unigramas y bigramas)
- Capa oculta 1: 256 neuronas, activacion ReLU, Dropout 0.3
- Capa oculta 2: 64 neuronas, activacion ReLU, Dropout 0.3
- Salida: 1 neurona con activacion Sigmoid (clasificacion binaria)

Configuracion de entrenamiento:
- Optimizador: Adam (lr=0.001)
- Funcion de perdida: Binary Cross-Entropy
- Hasta 20 epocas con early stopping (patience=5)
- Scheduler: ReduceLROnPlateau

Esta etapa incluye analisis exploratorio de datos (EDA) con distribucion de clases, analisis de longitud de resenas, nubes de palabras por sentimiento y un analisis de errores del modelo.

### Etapa 2 — Arquitectura Profunda: LSTM Bidireccional

Se reemplaza el MLP por una red recurrente capaz de capturar dependencias secuenciales en el texto, lo que el enfoque TF-IDF no puede representar.

Arquitectura del LSTM:
- Capa de embeddings entrenables (vocabulario de 25,000 tokens, dimension 128)
- LSTM Bidireccional con 2 capas y 128 unidades ocultas por direccion
- Dropout 0.3
- Cabeza de clasificacion: Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 1) -> Sigmoid

Configuracion de entrenamiento:
- Optimizador: Adam (lr=0.001) con gradient clipping (max_norm=1.0)
- Hasta 15 epocas con early stopping (patience=4)
- Padding dinamico por batch

Adicionalmente se realiza una visualizacion PCA 2D de los embeddings aprendidos para palabras de connotacion positiva y negativa.

### Etapa 3 — Modelos Preentrenados: DistilBERT y MiniLM

Se utilizan modelos Transformer preentrenados como extractores de caracteristicas (feature extractors), sin modificar sus pesos.

Modelos utilizados:
- DistilBERT (distilbert-base-uncased, 66M parametros): extrae embeddings CLS de cada resena.
- MiniLM (microsoft/MiniLM-L12-H384-uncased): modelo alternativo de menor tamano evaluado en paralelo.

Sobre los embeddings extraidos se entrena un clasificador Logistic Regression. Esta etapa demuestra el poder de la transferencia de conocimiento sin necesidad de fine-tuning.

### Etapa 4 — Componente Generativo: GPT-2 para Data Augmentation

Se incorpora un modelo generativo (GPT-2 small, 124M parametros) con el objetivo de producir resenas sinteticas que enriquezcan el conjunto de entrenamiento.

Metodologia:
- Generacion condicional mediante prompts especificos para sentimiento positivo y negativo
- 8 prompts por clase, 25 resenas por prompt
- Parametros de generacion: temperature=0.9, top_p=0.92, top_k=50, no_repeat_ngram_size=3

Experimento de Data Augmentation:
- Se simula un escenario de pocos datos (5,000 muestras de entrenamiento)
- Se compara el rendimiento de un clasificador TF-IDF + Logistic Regression con y sin las resenas generadas por GPT-2
- Las resenas generadas son validadas con un clasificador externo (distilbert-base-uncased-finetuned-sst-2-english)

### Etapa 5 — Fine-Tuning y Despliegue: DistilBERT + Gradio

Se realiza fine-tuning parcial de DistilBERT sobre el dataset IMDB y se despliega el modelo mediante una interfaz interactiva con Gradio.

Estrategia de fine-tuning parcial:
- Capas congeladas: embeddings + transformer layers 0 a 3
- Capas entrenables: transformer layers 4 y 5 + cabeza de clasificacion
- Solo el 20% aproximado de los parametros se actualizan durante el entrenamiento

Configuracion de entrenamiento:
- Subset de entrenamiento: 10,000 muestras / validacion: 2,000 muestras
- Optimizador: AdamW (lr=2e-5, weight_decay=0.01)
- Linear learning rate decay durante 3 epocas
- Gradient clipping (max_norm=1.0)
- Batch size: 16

El modelo final se guarda localmente y se sirve a traves de una interfaz Gradio con ejemplos predefinidos.

---

## Modelos Utilizados

| Modelo              | Parametros | Tipo                  |
|---------------------|------------|-----------------------|
| MLP                 | ~2.6M      | Feedforward           |
| LSTM Bidireccional  | ~3.5M      | Recurrente            |
| DistilBERT          | 66M        | Transformer (frozen)  |
| MiniLM-L12          | ~33M       | Transformer (frozen)  |
| GPT-2 small         | 124M       | Generativo            |
| DistilBERT FT       | 66M        | Transformer (tuned)   |

---

## Resultados por Etapa

Todas las etapas son evaluadas sobre el mismo conjunto de test con las siguientes metricas: Accuracy, Precision, Recall, F1-Score y AUC-ROC. La progresion de resultados ilustra la mejora obtenida a medida que se incorporan tecnicas mas avanzadas.

Etapa 1 (MLP + TF-IDF): linea base del proyecto.
Etapa 2 (LSTM Bidireccional): mejora al capturar dependencias de orden en el texto.
Etapa 3 (DistilBERT + LogReg): salto significativo gracias a los embeddings contextuales preentrenados.
Etapa 5 (DistilBERT Fine-Tuned): mejor resultado global del proyecto.

Los resultados numericos exactos se registran en el archivo resultados_finales.json generado al final de la ejecucion del notebook.

---

## Requisitos y Entorno

El proyecto fue desarrollado y ejecutado sobre la plataforma Kaggle con acelerador GPU (NVIDIA Tesla P100 o T4). La gestion de memoria GPU entre etapas es explicita para permitir la ejecucion secuencial en un mismo kernel.

Dependencias principales:
- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- Gradio
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- wordcloud
- tqdm

Instalacion adicional requerida en Kaggle:

    pip install wordcloud gradio

Todos los modelos de HuggingFace se descargan una vez al inicio del notebook con logica de reintentos y timeout, y quedan cacheados localmente. Desde ese punto, el modo offline queda activado y no se realizan mas llamadas de red.

---

## Instrucciones de Uso

1. Agregar el dataset IMDB como Input en Kaggle (buscar "imdb-dataset-of-50k-movie-reviews").
2. Abrir el notebook en Kaggle y activar el acelerador GPU desde la configuracion de la sesion.
3. Ejecutar todas las celdas en orden. La celda de descarga de modelos puede tardar varios minutos en la primera ejecucion.
4. Al finalizar la Etapa 5, Gradio generara un enlace publico temporal para probar el modelo en el navegador.

Para ejecutar localmente (fuera de Kaggle), ajustar las rutas de los datasets y modelos segun corresponda, y asegurarse de contar con una GPU con al menos 12 GB de VRAM para las etapas de Transformer.

---

## Estructura del Repositorio

    .
    |-- Proyecto_Final_DL_Kaggle-ejecutado.ipynb   Notebook principal ejecutado con todas las etapas
    |-- resultados_finales.json                    Metricas de todas las etapas en formato JSON
    |-- README.md                                  Este archivo

---

## Plataforma de Ejecucion

- Kaggle Notebooks
- GPU: NVIDIA Tesla P100 (16 GB) o T4 x2
- El notebook gestiona automaticamente la deteccion del tipo de GPU y adapta la configuracion de precision numerica (FP16/FP32) en consecuencia.

---

## Notas Academicas

Este proyecto fue desarrollado como trabajo final de la asignatura de Deep Learning en el marco de una Maestria en Ciencia de Datos. El objetivo es demostrar la progresion desde tecnicas clasicas de representacion de texto hasta el uso de modelos Transformer de gran escala, incluyendo un componente generativo con vinculo directo al problema de clasificacion.
