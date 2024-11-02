import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import shutil  # Importar shutil para eliminar directorios

def get_base_path():
    # Obtiene la ruta base dependiendo del entorno
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # Ruta del ejecutable PyInstaller
    return os.path.abspath(".")

def train_model(data_path):
    # Inicialización de variables globales
    modelo = None
    vectorizer = None

    # Definir las rutas para el modelo y el vectorizador en la carpeta custom_model
    base_path = get_base_path()
    model_dir = os.path.join(base_path, 'custom_model')
    
    # Verificar si la carpeta custom_model existe
    if os.path.exists(model_dir):
        # Si existe, eliminar su contenido
        shutil.rmtree(model_dir)
        print("Contenido de 'custom_model' eliminado.")

    # Crear la carpeta custom_model
    os.makedirs(model_dir)
    print("Carpeta 'custom_model' creada.")

    model_path = os.path.join(model_dir, 'text_classifier_model.joblib')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')

    print("Entrenando nuevo modelo...")

    # Cargar datos desde el archivo JSON
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error al cargar el archivo de datos: {e}")
        return None, None

    # Preparar datos para el DataFrame
    textos = []
    etiquetas = []

    for tipo, ejemplos in data['data'].items():
        for ejemplo in ejemplos:
            textos.append(ejemplo)
            etiquetas.append(tipo)

    # Crear un DataFrame
    df = pd.DataFrame({'texto': textos, 'etiqueta': etiquetas})

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df['texto'], df['etiqueta'], test_size=0.2, random_state=42)

    # Vectorizar los textos
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Crear y entrenar el clasificador
    modelo = MultinomialNB()
    modelo.fit(X_train_vectorized, y_train)

    # Predecir en el conjunto de prueba
    y_pred = modelo.predict(vectorizer.transform(X_test))

    # Evaluar el modelo
    print("Modelo entrenado con éxito.")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Guardar el modelo y el vectorizador
    joblib.dump(modelo, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return modelo, vectorizer

if __name__ == "__main__":
    print("Este script se debe importar en app.py para ejecutar el entrenamiento del modelo.")
