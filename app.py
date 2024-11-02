from tkinter import Tk, Text, Button, Label, Frame, Listbox, END, filedialog, messagebox, Menu, Toplevel
import os
import joblib
import sys
import train  # Importar el módulo train

# Configuración inicial de la aplicación
modelo = None
vectorizer = None
modelo_activo = "model"  # Usar por defecto la carpeta 'model'
fecha_actualizacion = "02/11/2024"  # Fecha de actualización del modelo nativo

# Función para obtener la ruta base dependiendo del entorno
def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # Ruta del ejecutable PyInstaller
    return os.path.abspath(".")

# Función para cargar el modelo y el vectorizador
def load_model():
    global modelo, vectorizer
    base_path = get_base_path()
    model_path = os.path.join(base_path, modelo_activo, 'text_classifier_model.joblib')
    vectorizer_path = os.path.join(base_path, modelo_activo, 'vectorizer.joblib')

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Cargando modelo y vectorizador desde:", modelo_activo)
        modelo = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Modelo y vectorizador cargados con éxito.")
    else:
        raise FileNotFoundError("No se encontraron el modelo o el vectorizador en la carpeta seleccionada.")

# Función para clasificar el texto ingresado
def clasificar_texto():
    texto = texto_area.get("1.0", END).strip()
    if not texto:
        resultado_label.config(text="Por favor, introduce un texto.")
        return

    # Vectorización y predicción
    texto_vectorizado = vectorizer.transform([texto])
    prediccion = modelo.predict(texto_vectorizado)
    probabilidades = modelo.predict_proba(texto_vectorizado).flatten()
    
    # Mostrar la clase
    resultado_label.config(text=f"El texto es de tipo: {prediccion[0]}")

    # Limpiar lista de probabilidades anteriores
    probabilidades_lista.delete(0, END)

    # Mostrar las probabilidades
    clases = modelo.classes_
    for clase, probabilidad in zip(clases, probabilidades):
        probabilidades_lista.insert(END, f"{clase}: {probabilidad * 100:.2f}%")

# Función para usar un modelo personalizado
def usar_modelo_personalizado():
    respuesta = messagebox.askyesno("Confirmar", "¿Quieres usar un modelo personalizado?")

    if respuesta:
        archivo = filedialog.askopenfilename(title="Seleccionar archivo data.json", filetypes=[("JSON files", "*.json")])
        if archivo:
            try:
                # Llamar a la función de entrenamiento con el archivo seleccionado
                train.train_model(archivo)  # Asegúrate de que train.py pueda recibir el archivo
                global modelo_activo  # Permitir que esta función cambie el modelo activo
                modelo_activo = "custom_model"  # Cambiar a usar la carpeta 'custom_model'
                load_model()  # Cargar el nuevo modelo entrenado
                messagebox.showinfo("Éxito", "Modelo entrenado y cargado con éxito.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")

# Función para mostrar información del modelo
def mostrar_info_modelo():
    info_ventana = Toplevel(ventana)
    info_ventana.title("Info del Modelo")
    info_ventana.geometry("300x150")
    
    if modelo_activo == "model":
        texto_info = f"Modelo en uso: Nativo\nÚltima fecha de actualización: {fecha_actualizacion}"
    else:
        texto_info = "Modelo en uso: Personalizado"

    label_info = Label(info_ventana, text=texto_info, padx=10, pady=10)
    label_info.pack()

# Crear la interfaz de usuario
ventana = Tk()
ventana.title("Clasificador de Texto")
ventana.geometry("600x600")  # Aumentar la altura de la ventana
ventana.config(bg="#f4f4f4")

# Contenedor para el texto y el botón
contenedor = Frame(ventana, bg="white", padx=20, pady=20)
contenedor.pack(padx=20, pady=20, fill="both", expand=True)

titulo_label = Label(contenedor, text="Clasificador de Texto", font=("Arial", 20, "bold"), fg="#333", bg="white")
titulo_label.pack(pady=(0, 10))

texto_label = Label(contenedor, text="Introduce el texto para clasificar:", font=("Arial", 12, "bold"), bg="white")
texto_label.pack(anchor="w")

texto_area = Text(contenedor, height=10, font=("Arial", 14), wrap="word", padx=10, pady=10, relief="solid")
texto_area.pack(fill="x", pady=(0, 10))

# Botón para clasificar texto
clasificar_btn = Button(contenedor, text="Clasificar Texto", font=("Arial", 14), bg="#007BFF", fg="white", relief="flat", command=clasificar_texto)
clasificar_btn.pack(fill="x", pady=(0, 10))

resultado_label = Label(contenedor, font=("Arial", 14), fg="#444", bg="white")
resultado_label.pack(pady=(10, 5))

# Contenedor para la lista de probabilidades
probabilidades_label = Label(contenedor, text="Probabilidades:", font=("Arial", 12, "bold"), bg="white")
probabilidades_label.pack(anchor="w")

# Usamos un Listbox para mostrar las probabilidades
probabilidades_lista = Listbox(contenedor, font=("Arial", 12), height=3, bg="#f9f9f9", bd=2, relief="groove")
probabilidades_lista.pack(fill="both", expand=True)

# Menú contextual
def mostrar_menu(event):
    menu.post(event.x_root, event.y_root)

menu = Menu(ventana, tearoff=0)
menu.add_command(label="Usar modelo personalizado", command=usar_modelo_personalizado)
menu.add_command(label="Info del Modelo", command=mostrar_info_modelo)  # Nueva opción añadida

# Conectar el clic derecho al menú contextual
ventana.bind("<Button-3>", mostrar_menu)

# Cargar el modelo al iniciar la aplicación
try:
    load_model()
    ventana.mainloop()
except FileNotFoundError as e:
    print(e)
