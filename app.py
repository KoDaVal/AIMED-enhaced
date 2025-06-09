from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import process
from pytrends.request import TrendReq

app = Flask(__name__)
pytrends = TrendReq(hl="es-MX", tz=360)

# 📌 Base de datos de enfermedades MEJORADA
data = {
    "Enfermedad": [
        "Gripe", "Covid-19", "Alergia", "Gastroenteritis", "Bronquitis",
        "Neumonía", "Infarto", "Dengue"
    ],
    "Síntomas": [
        ["fiebre", "tos", "dolor de cabeza", "estornudos", "dolor de garganta", "escalofríos", "dolor muscular", "fatiga", "congestión nasal", "secreción nasal"], # Gripe
        ["fiebre", "tos seca", "perdida del olfato", "perdida del gusto", "dificultad para respirar", "fatiga", "dolor de garganta", "dolor de cabeza", "dolor muscular", "escalofríos", "congestión nasal", "náuseas", "vómitos", "diarrea", "erupción en la piel", "dolor en el pecho", "falta de aire"], # Covid-19
        ["estornudos", "ojos rojos", "picazón", "congestión nasal", "secreción nasal", "lagrimeo", "irritación de ojos", "picazón de garganta"], # Alergia
        ["diarrea", "dolor de estómago", "náuseas", "vómitos", "dolor abdominal", "calambres abdominales", "pérdida de apetito", "deshidratación"], # Gastroenteritis
        ["tos seca", "tos con flema", "dolor en el pecho", "fatiga", "dificultad para respirar", "sibilancias", "opresión en el pecho", "fiebre leve", "escalofríos"], # Bronquitis
        ["fiebre alta", "escalofríos", "dificultad para respirar", "dolor al respirar", "tos con flema", "tos productiva", "dolor en el pecho", "fatiga", "sudoración", "confusión (en ancianos)"], # Neumonía
        ["dolor en el pecho", "sudor frío", "mareos", "náuseas", "Falta de aire", "dolor en el brazo izquierdo", "dolor en la mandíbula", "dolor en la espalda", "malestar en el pecho", "presión en el pecho", "ardor en el pecho"], # Infarto
        ["fiebre alta", "dolor muscular", "erupción en la piel", "dolor en las articulaciones", "dolor detrás de los ojos", "cansancio extremo", "náuseas", "vómitos", "sangrado leve (encías, nariz)"] # Dengue
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

df = pd.DataFrame(data)
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Síntomas"])
y = df["Enfermedad"]

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

def corregir_sintomas(sintomas_usuario):
    sintomas_usuario = sintomas_usuario.lower().split(", ")
    # Validación con umbral para evitar falsos positivos:
    sintomas_corregidos = []
    for s in sintomas_usuario:
        mejor, score = process.extractOne(s, mlb.classes_)
        if score >= 70:
            sintomas_corregidos.append(mejor)
    return sintomas_corregidos

def verificar_tendencia_google(enfermedad, ubicacion):
    try:
        region = "MX" if "México" in ubicacion else ""
        pytrends.build_payload([enfermedad], geo=region, timeframe="today 3-m")
        data = pytrends.interest_over_time()

        if not data.empty and data[enfermedad].sum() > 0:
            return f"📊 {enfermedad} ha sido tendencia en {ubicacion} recientemente."
        else:
            return f"📉 No hay tendencias recientes sobre {enfermedad} en {ubicacion}."
    except:
        return f"⚠ No se pudo obtener información de Google Trends para {enfermedad}."

def calcular_imc(peso, altura):
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
    # Lista de enfermedades a monitorear
    enfermedades = ["Covid-19", "Dengue", "Gripe", "Neumonía"]
    brotes = []
    try:
        for enfermedad in enfermedades:
            pytrends.build_payload([enfermedad], geo="", timeframe="now 7-d")
            tendencias = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
            if not tendencias.empty:
                # Se obtienen las ciudades con mayor interés por la enfermedad
                tendencias = tendencias.sort_values(by=enfermedad, ascending=False)
                top_ciudades = [c.lower() for c in tendencias.head(20).index]
                if ubicacion.lower() in top_ciudades:
                    brotes.append(enfermedad)
    except:
        pass
    return brotes

def detectar_brotes_viajes(viajes):
    destinos = [v.strip().lower() for v in viajes if v.strip().lower() != "ninguno"]
    brotes_detectados = {}
    for destino in destinos:
        brotes = detectar_brotes_ubicacion(destino)
        if brotes:
            brotes_detectados[destino] = brotes
    return brotes_detectados
# -------------------------------------------------------

def diagnosticar(nombre, sintomas_usuario, edad, sexo, peso, altura, ubicacion, viajes):
    sintomas_usuario = corregir_sintomas(sintomas_usuario)
    sintomas_numericos = mlb.transform([sintomas_usuario])

    try:
        enfermedad_predicha = modelo.predict(sintomas_numericos)[0]
        enfermedad_info = df[df["Enfermedad"] == enfermedad_predicha].iloc[0]
        tendencia_google = verificar_tendencia_google(enfermedad_predicha, ubicacion)
        estado_peso = calcular_imc(peso, altura)

        # --- Detección de brotes por ubicación y viajes ---
        brotes_en_ubicacion = detectar_brotes_ubicacion(ubicacion)
        brotes_por_viajes = detectar_brotes_viajes(viajes)

        alertas_brotes = ""
        if brotes_en_ubicacion:
            alertas_brotes += f"📍 En {ubicacion.title()} hay tendencia reciente de: {', '.join(brotes_en_ubicacion)}.\n"
        if brotes_por_viajes:
            for lugar, enfermedades in brotes_por_viajes.items():
                alertas_brotes += f"✈️ En tu destino reciente '{lugar.title()}' se reporta: {', '.join(enfermedades)}.\n"
        # ----------------------------------------------------

        emergencia = "🔴 ¡Emergencia médica! 🚨" if enfermedad_info["Emergencia"] else "🟢 No es emergencia inmediata."

        return (
            f"👤 {nombre}, aquí está tu diagnóstico:\n"
            f"📝 Síntomas reconocidos: {', '.join(sintomas_usuario)}\n"
            f"{alertas_brotes}"
            f"📌 Enfermedad probable: {enfermedad_predicha}\n"
            f"ℹ {enfermedad_info['Descripcion']}\n"
            f"💊 Recomendación: {enfermedad_info['Recomendacion']}\n"
            f"{estado_peso}\n"
            f"{tendencia_google}\n"
            f"{emergencia}"
        )
    except Exception as e: # Captura la excepción para verla
        print(f"Error en diagnosticar: {e}") # Imprime el error en los logs
        return f"⚠ {nombre}, no se encontró una coincidencia exacta o hubo un error en el diagnóstico. Consulta a un médico. (Error: {e})" # Mensaje más detallado para debug

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    # Mueve la definición de sintomas_disponibles aquí, fuera del bloque POST
    # para que esté disponible para solicitudes GET
    sintomas_disponibles = sorted(list(mlb.classes_))

    if request.method == "POST":
        nombre = request.form["nombre"]
        sintomas = request.form["sintomas"]
        edad = int(request.form["edad"])
        sexo = request.form["sexo"]
        peso = float(request.form["peso"])
        altura = float(request.form["altura"])
        ubicacion = request.form["ubicacion"]
        viajes = request.form["viajes"].split(", ") if request.form["viajes"] else []

        resultado = diagnosticar(nombre, sintomas, edad, sexo, peso, altura, ubicacion, viajes)

    # Pasa la lista de síntomas a la plantilla
    return render_template("index.html", resultado=resultado, sintomas_disponibles=sintomas_disponibles)

if __name__ == "__main__":
    app.run(debug=True)
