from flask import Flask, request, jsonify, render_template
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

app = Flask(__name__, template_folder='.')

# Inicialización del modelo y el vectorizador
modelo = None
vectorizer = None

def train_model():
    global modelo, vectorizer  # Permitir que estas variables sean accesibles fuera de la función
    
    # Definir las rutas para el modelo y el vectorizador
    model_path = 'model/text_classifier_model.joblib'
    vectorizer_path = 'model/vectorizer.joblib'
    
    # Comprobar si el modelo y el vectorizador ya existen
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Modelo y vectorizador ya existentes. Cargando...")
        modelo = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return

    print("Entrenando nuevo modelo...")

    # Cargar datos desde el archivo JSON
    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

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

@app.route('/')
def home():
    return render_template('index.html')  # Asegúrate de que el archivo esté en la raíz ahora

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    texto = data['texto']
    
    # Vectorizar el texto
    texto_vectorizado = vectorizer.transform([texto])
    
    # Predecir la clase y las probabilidades
    prediccion = modelo.predict(texto_vectorizado)
    probabilidades = modelo.predict_proba(texto_vectorizado).flatten()
    
    # Crear un diccionario para las probabilidades
    clases = modelo.classes_
    probabilidades_dict = {clase: probabilidad for clase, probabilidad in zip(clases, probabilidades)}
    
    return jsonify({
        'clase': prediccion[0],
        'probabilidades': probabilidades_dict
    })

if __name__ == '__main__':
    train_model()  # Entrena el modelo al iniciar la aplicación
    app.run(debug=True)
