from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import process
from pytrends.request import TrendReq

app = Flask(__name__)
pytrends = TrendReq(hl="es-MX", tz=360)

# 📌 Base de datos de enfermedades MEJORADA
# Esta base de datos ha sido ampliada para incluir más síntomas distintivos
# y variaciones comunes que ayudan al modelo a diferenciar mejor las enfermedades.
data = {
    "Enfermedad": [
        "Gripe", "Covid-19", "Alergia", "Gastroenteritis", "Bronquitis",
        "Neumonía", "Infarto", "Dengue"
    ],
    "Síntomas": [
        ["fiebre", "tos", "dolor de cabeza", "estornudos", "dolor de garganta", "escalofríos", "dolor muscular", "fatiga", "congestión nasal", "secreción nasal", "malestar general"], # Gripe
        ["fiebre", "tos seca", "perdida del olfato", "perdida del gusto", "dificultad para respirar", "fatiga", "dolor de garganta", "dolor de cabeza", "dolor muscular", "escalofríos", "congestión nasal", "náuseas", "vómitos", "diarrea", "erupción en la piel", "dolor en el pecho", "falta de aire", "dolor articular"], # Covid-19
        ["estornudos", "ojos rojos", "picazón", "congestión nasal", "secreción nasal", "lagrimeo", "irritación de ojos", "picazón de garganta", "picazón de nariz", "urticaria", "erupciones cutaneas"], # Alergia
        ["diarrea", "dolor de estómago", "náuseas", "vómitos", "dolor abdominal", "calambres abdominales", "pérdida de apetito", "deshidratación", "fiebre leve", "malestar estomacal"], # Gastroenteritis
        ["tos seca", "tos con flema", "dolor en el pecho", "fatiga", "dificultad para respirar", "sibilancias", "opresión en el pecho", "fiebre leve", "escalofríos", "cansancio", "respiración ruidosa"], # Bronquitis
        ["fiebre alta", "escalofríos", "dificultad para respirar", "dolor al respirar", "tos con flema", "tos productiva", "dolor en el pecho", "fatiga", "sudoración", "confusión (en ancianos)", "cianosis", "esputo con sangre"], # Neumonía
        ["dolor en el pecho", "sudor frío", "mareos", "náuseas", "Falta de aire", "dolor en el brazo izquierdo", "dolor en la mandíbula", "dolor en la espalda", "malestar en el pecho", "presión en el pecho", "ardor en el pecho", "desmayo", "palpitaciones"], # Infarto
        ["fiebre alta", "dolor muscular", "erupción en la piel", "dolor en las articulaciones", "dolor detrás de los ojos", "cansancio extremo", "náuseas", "vómitos", "sangrado leve (encías, nariz)", "dolor de huesos", "petequias", "cefalea"] # Dengue
    ],
    "Emergencia": [False, False, False, False, False, True, True, False],
    "Descripcion": [
        "La gripe es una infección viral que causa fiebre, tos y dolor muscular.",
        "El Covid-19 es una enfermedad respiratoria con fiebre, tos seca y fatiga. Parecido a una gripe pero con perdida de sentidos",
        "Las alergias son respuestas inmunológicas a sustancias como polvo o polen, o algún alimento.",
        "La gastroenteritis es una inflamación del estómago provocando diarrea y vómitos.",
        "La bronquitis es una inflamación de los bronquios, causando tos persistente. si esta dura mas de 3 semanas puede ser bronquitis crónica",
        "La neumonía es una infección pulmonar grave con fiebre alta y dificultad respiratoria.",
        "El infarto ocurre cuando el flujo sanguíneo al corazón se bloquea.",
        "El dengue es una infección viral transmitida por mosquitos que causa fiebre y dolores intensos.",
    ],
    "Recomendacion": [
        "Tomar paracetamol, reposo y líquidos. ⚠️ Si la fiebre dura mas de 3 días o supera los 39 grados consulta a tu médico",
        "Aislamiento para evitar contagios y tomar paracetamol. ⚠️ Consultar a un médico si hay dificultad para respirar o coloracion azulada en labios o piel.",
        "Evitar alérgenos y tomar antihistamínicos. ⚠️ Consultar al medico si hay dificultad para respirar o hinchazon en cara y garganta.",
        "Beber líquidos y seguir una dieta blanda. ⚠️ consultar al medico si la diarrea es persistente, o si el vomito impide la hidratación",
        "Descansar y consultar a un médico si la tos persiste.",
        "⚠️Ir al hospital por un chequeo medico.",
        " ⚠️ ¡Llamar al 911 o acudir a emergencias inmediatamente! ⚠️",
        "Reposo, hidratación, análisis de sangre y acudir al médico si hay síntomas graves.",
    ]
}

# Creación del DataFrame de Pandas
df = pd.DataFrame(data)

# Inicialización y ajuste del MultiLabelBinarizer con los síntomas de la base de datos
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Síntomas"]) # 'mlb.classes_' contendrá todos los síntomas únicos del dataset
y = df["Enfermedad"]

# Entrenamiento del modelo de Random Forest Classifier
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

def corregir_sintomas(sintomas_usuario_input_str):
    """
    Procesa la cadena de síntomas ingresados por el usuario, intentando
    encontrar coincidencias con los síntomas conocidos por el modelo de IA.
    Solo los síntomas con un score de coincidencia suficiente serán "reconocidos"
    y se usarán para el diagnóstico interno de la IA.
    """
    sintomas_usuario_parsed = sintomas_usuario_input_str.lower().split(", ")
    sintomas_reconocidos_para_ia = [] # Lista que contendrá los síntomas que la IA 'entiende'
    
    for s_input in sintomas_usuario_parsed:
        s_input_cleaned = s_input.strip()
        if not s_input_cleaned: # Saltar entradas vacías si hay comas adicionales
            continue

        # Busca la mejor coincidencia del síntoma del usuario en la base de datos de síntomas del modelo
        mejor_coincidencia, score = process.extractOne(s_input_cleaned, mlb.classes_)
        
        # Umbral de coincidencia: si el score es bajo, el síntoma no se considera "reconocido" por la IA.
        # Ajustado a 60 para ser un poco menos estricto y permitir que más síntomas contribuyan al diagnóstico.
        if score >= 60: 
            sintomas_reconocidos_para_ia.append(mejor_coincidencia)
        # Puedes añadir aquí un 'else' para depurar si un síntoma no es reconocido por la IA
        # else:
            # print(f"DEBUG (corregir_sintomas): '{s_input_cleaned}' NO fue reconocido por la IA (score: {score}) - Mejor coincidencia en la base de datos: '{mejor_coincidencia}'")

    return sintomas_reconocidos_para_ia

def verificar_tendencia_google(enfermedad, ubicacion):
    """
    Verifica si una enfermedad es tendencia en Google Trends para una ubicación específica.
    """
    try:
        region = "MX" if "México" in ubicacion else "" # Simplificación: si contiene "México", asume región MX
        pytrends.build_payload([enfermedad], geo=region, timeframe="today 3-m")
        data = pytrends.interest_over_time()
        
        if not data.empty and data[enfermedad].sum() > 0:
            return f"📊 {enfermedad} ha sido tendencia en {ubicacion} recientemente."
        else:
            return f"📉 No hay tendencias recientes sobre {enfermedad} en {ubicacion}."
    except Exception as e: # Captura la excepción para evitar que falle toda la app
        print(f"Error al verificar tendencia de Google para {enfermedad}: {e}")
        return f"⚠ No se pudo obtener información de Google Trends para {enfermedad}."

def calcular_imc(peso, altura):
    """
    Calcula el Índice de Masa Corporal (IMC) y lo clasifica.
    Peso en kg, Altura en cm.
    """
    if altura == 0: # Evitar división por cero
        return "IMC: Error - Altura no puede ser cero."
    imc = peso / ((altura / 100) ** 2)
    if imc < 18.5:
        return f"IMC: {imc:.2f} - Bajo peso"
    elif 18.5 <= imc < 24.9:
        return f"IMC: {imc:.2f} - Peso ideal"
    elif 25 <= imc < 29.9:
        return f"IMC: {imc:.2f} - Sobrepeso"
    else:
        return f"IMC: {imc:.2f} - Obesidad"

# --- NUEVAS FUNCIONES: Detección de brotes por ubicación y viajes ---
def detectar_brotes_ubicacion(ubicacion):
    """
    Detecta tendencias de brotes de enfermedades específicas en una ubicación.
    Utiliza Google Trends a nivel de ciudad.
    """
    enfermedades_a_monitorear = ["Covid-19", "Dengue", "Gripe", "Neumonía"]
    brotes_detectados = []
    try:
        for enfermedad in enfermedades_a_monitorear:
            # Para Google Trends a nivel de ciudad, 'geo' debe estar vacío para buscar por resolución de ciudad
            pytrends.build_payload([enfermedad], geo="", timeframe="now 7-d")
            # inc_low_vol=True para incluir datos de bajo volumen, crucial para ciudades más pequeñas
            tendencias = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
            
            if not tendencias.empty:
                # Obtiene las ciudades con mayor interés por la enfermedad (top 20)
                tendencias_ordenadas = tendencias.sort_values(by=enfermedad, ascending=False)
                top_ciudades = [c.lower() for c in tendencias_ordenadas.head(20).index]
                
                # Comprueba si la ubicación del usuario está en el top de ciudades con tendencia
                if ubicacion.lower() in top_ciudades:
                    brotes_detectados.append(enfermedad)
    except Exception as e:
        print(f"Error al detectar brotes en {ubicacion}: {e}")
        pass # Ignorar errores de pytrends para no detener el diagnóstico
    return brotes_detectados

def detectar_brotes_viajes(viajes):
    """
    Detecta brotes en los destinos de viaje recientes del usuario.
    """
    destinos = [v.strip().lower() for v in viajes if v.strip().lower() != "ninguno"]
    brotes_por_viajes_dict = {}
    for destino in destinos:
        brotes = detectar_brotes_ubicacion(destino)
        if brotes:
            brotes_por_viajes_dict[destino] = brotes
    return brotes_por_viajes_dict
# -------------------------------------------------------

def diagnosticar(nombre, sintomas_usuario_raw_str, edad, sexo, peso, altura, ubicacion, viajes):
    """
    Realiza el diagnóstico completo basándose en los síntomas y datos del usuario.
    """
    # Procesa y reconoce los síntomas para el modelo de IA.
    # Esta lista (sintomas_para_el_modelo_ia) es la que se usa para la predicción.
    sintomas_para_el_modelo_ia = corregir_sintomas(sintomas_usuario_raw_str)
    
    # Transforma los síntomas reconocidos a formato numérico para el modelo
    sintomas_numericos = mlb.transform([sintomas_para_el_modelo_ia])

    # Convertir la cadena de síntomas original del usuario a una lista para mostrar
    # Esto es lo que el usuario vio en el formulario, independientemente de si la IA lo "reconoció".
    sintomas_ingresados_por_usuario = [s.strip() for s in sintomas_usuario_raw_str.split(',') if s.strip()]


    try:
        # Realiza la predicción de la enfermedad
        enfermedad_predicha = modelo.predict(sintomas_numericos)[0]
        
        # Obtiene la información detallada de la enfermedad predicha
        enfermedad_info = df[df["Enfermedad"] == enfermedad_predicha].iloc[0]
        
        # Obtiene información adicional
        tendencia_google = verificar_tendencia_google(enfermedad_predicha, ubicacion)
        estado_peso = calcular_imc(peso, altura)
        
        # Detección de brotes
        brotes_en_ubicacion = detectar_brotes_ubicacion(ubicacion)
        brotes_por_viajes = detectar_brotes_viajes(viajes)

        # Formatea las alertas de brotes
        alertas_brotes = ""
        if brotes_en_ubicacion:
            alertas_brotes += f"📍 En {ubicacion.title()} hay tendencia reciente de: {', '.join(brotes_en_ubicacion)}.\n"
        if brotes_por_viajes:
            for lugar, enfermedades in brotes_por_viajes.items():
                alertas_brotes += f"✈️ En tu destino reciente '{lugar.title()}' se reporta: {', '.join(enfermedades)}.\n"
        
        # Determina si es una emergencia
        emergencia = "🔴 ¡Emergencia médica! 🚨" if enfermedad_info["Emergencia"] else "🟢 No es emergencia inmediata."
        
        # Construye el mensaje de resultado final
        return (
            f"👤 {nombre}, aquí está tu diagnóstico:\n"
            # Muestra los síntomas EXACTOS que el usuario ingresó en el formulario.
            f"📝 Síntomas ingresados: {', '.join(sintomas_ingresados_por_usuario) if sintomas_ingresados_por_usuario else 'Ninguno'}\n"
            f"{alertas_brotes}"
            f"📌 Enfermedad probable: {enfermedad_predicha}\n"
            f"ℹ {enfermedad_info['Descripcion']}\n"
            f"💊 Recomendación: {enfermedad_info['Recomendacion']}\n"
            f"{estado_peso}\n"
            f"{tendencia_google}\n"
            f"{emergencia}"
        )
    except Exception as e:
        # Captura cualquier error inesperado durante el diagnóstico y lo imprime en los logs
        print(f"Error inesperado en la función diagnosticar: {e}")
        return f"⚠ {nombre}, no se pudo completar el diagnóstico debido a un error interno. Por favor, consulta a un médico. (Detalle: {e})"

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Ruta principal de la aplicación Flask.
    Maneja las solicitudes GET (para cargar la página) y POST (para enviar el formulario).
    """
    resultado = ""
    # Esta línea se ha movido fuera del bloque POST para asegurar que 'sintomas_disponibles'
    # siempre esté definida y pueda pasarse a la plantilla, incluso en una solicitud GET inicial.
    sintomas_disponibles = sorted(list(mlb.classes_))

    if request.method == "POST":
        # Obtiene los datos del formulario
        nombre = request.form["nombre"]
        # 'sintomas' se recibe como una cadena separada por comas desde el campo oculto
        sintomas_raw_del_formulario = request.form["sintomas"] 
        edad = int(request.form["edad"])
        sexo = request.form["sexo"]
        peso = float(request.form["peso"])
        altura = float(request.form["altura"])
        ubicacion = request.form["ubicacion"]
        viajes = request.form["viajes"].split(", ") if request.form["viajes"] else []

        # Llama a la función de diagnóstico. 
        # NOTA: Le pasamos los síntomas RAW del formulario (sintomas_raw_del_formulario)
        # La función diagnosticar se encargará de pasarlos a corregir_sintomas INTERNAMENTE
        # para la IA, pero usará la versión RAW para mostrar en el resultado.
        resultado = diagnosticar(nombre, sintomas_raw_del_formulario, edad, sexo, peso, altura, ubicacion, viajes)

    # Renderiza la plantilla index.html, pasando el resultado del diagnóstico
    # y la lista de síntomas disponibles para el selector de Choices.js.
    return render_template("index.html", resultado=resultado, sintomas_disponibles=sintomas_disponibles)

if __name__ == "__main__":
    # Ejecuta la aplicación Flask en modo de depuración.
    # 'debug=True' permite recarga automática y muestra errores detallados,
    # ideal para desarrollo local. No usar en producción.
    app.run(debug=True)
