# Clasificador de Texto

Este proyecto es una aplicación web de clasificador de texto construida con Flask y utiliza un modelo de aprendizaje automático para clasificar textos en diferentes categorías. La aplicación permite a los usuarios ingresar texto, y el clasificador devuelve la categoría correspondiente junto con las probabilidades de cada clase.

## Tabla de Contenidos
- [Características](#características)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Configuración](#configuración)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Características
- **Clasificación de Texto**: Clasifica textos en diferentes categorías utilizando un modelo de Naive Bayes.
- **Interfaz de Usuario**: Una interfaz sencilla y clara para ingresar texto y ver resultados.
- **Entrenamiento Automático**: El modelo se entrena automáticamente si no se encuentra un modelo preexistente.
- **Resultados Detallados**: Muestra la clase predicha y las probabilidades asociadas a cada clase.

## Tecnologías Utilizadas
- [Flask](https://flask.palletsprojects.com/) - Un micro framework para Python.
- [Scikit-learn](https://scikit-learn.org/) - Biblioteca de aprendizaje automático en Python.
- [Pandas](https://pandas.pydata.org/) - Biblioteca para el manejo y análisis de datos.
- [Joblib](https://joblib.readthedocs.io/en/latest/) - Utilizado para guardar y cargar modelos de aprendizaje automático.
- [HTML/CSS](https://www.w3.org/TR/html52/) - Para la creación de la interfaz de usuario.


## Configuración
1. **Instalar Dependencias**: Asegúrate de tener `pip` instalado con las dependencias necesarias
2. **Ejectuar Aplicación**: Ejecuta app.py con el run.bat