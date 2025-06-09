from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import process
from pytrends.request import TrendReq

app = Flask(__name__)
pytrends = TrendReq(hl="es-MX", tz=360)

# 📌 Base de datos de enfermedades - SÍNTOMAS Y RECOMENDACIONES MEJORADAS
# Se han ampliado las listas de síntomas para una mayor cobertura y diferenciación.
# Las recomendaciones se han revisado para ser más detalladas y médicamente apropiadas.
data = {
    "Enfermedad": [
        "Gripe", "Covid-19", "Alergia", "Gastroenteritis", "Bronquitis", 
        "Neumonía", "Infarto", "Dengue"
    ],
    "Síntomas": [
        ["fiebre", "tos", "dolor de cabeza", "estornudos", "dolor de garganta", "escalofríos", "dolor muscular", "fatiga", "congestión nasal", "secreción nasal", "malestar general", "dolor de cuerpo", "voz ronca"], # Gripe
        ["fiebre", "tos seca", "perdida del olfato", "perdida del gusto", "dificultad para respirar", "fatiga", "dolor de garganta", "dolor de cabeza", "dolor muscular", "escalofríos", "congestión nasal", "náuseas", "vómitos", "diarrea", "erupción en la piel", "dolor en el pecho", "falta de aire", "dolor articular", "confusión", "dolor abdominal"], # Covid-19
        ["estornudos", "ojos rojos", "picazón", "congestión nasal", "secreción nasal", "lagrimeo", "irritación de ojos", "picazón de garganta", "picazón de nariz", "urticaria", "erupciones cutaneas", "hinchazón de cara", "sibilancias"], # Alergia
        ["diarrea", "dolor de estómago", "náuseas", "vómitos", "dolor abdominal", "calambres abdominales", "pérdida de apetito", "deshidratación", "fiebre leve", "malestar estomacal", "debilidad", "dolor de cabeza", "escalofríos"], # Gastroenteritis
        ["tos seca", "tos con flema", "dolor en el pecho", "fatiga", "dificultad para respirar", "sibilancias", "opresión en el pecho", "fiebre leve", "escalofríos", "cansancio", "respiración ruidosa", "dolor de garganta", "malestar general"], # Bronquitis
        ["fiebre alta", "escalofríos", "dificultad para respirar", "dolor al respirar", "tos con flema", "tos productiva", "dolor en el pecho", "fatiga", "sudoración", "confusión (en ancianos)", "cianosis", "esputo con sangre", "dolor de cabeza", "dolor muscular", "pérdida de apetito"], # Neumonía
        ["dolor en el pecho", "sudor frío", "mareos", "náuseas", "Falta de aire", "dolor en el brazo izquierdo", "dolor en la mandíbula", "dolor en la espalda", "malestar en el pecho", "presión en el pecho", "ardor en el pecho", "desmayo", "palpitaciones", "dolor en el cuello", "debilidad"], # Infarto
        ["fiebre alta", "dolor muscular", "erupción en la piel", "dolor en las articulaciones", "dolor detrás de los ojos", "cansancio extremo", "náuseas", "vómitos", "sangrado leve (encías, nariz)", "dolor de huesos", "petequias", "cefalea", "fatiga severa", "pérdida de apetito", "ganglios linfáticos inflamados"] # Dengue
    ],
    "Emergencia": [False, False, False, False, False, True, True, False],
    "Descripcion": [
        "La gripe es una infección viral respiratoria que causa fiebre, tos, dolor muscular y congestión. Es estacional y se transmite fácilmente.",
        "El Covid-19 es una enfermedad respiratoria causada por el virus SARS-CoV-2, con síntomas variables que pueden incluir fiebre, tos seca, pérdida del olfato/gusto y dificultad para respirar.",
        "Las alergias son respuestas exageradas del sistema inmunitario a sustancias inofensivas (alérgenos) como polen, polvo, alimentos o picaduras de insectos, manifestándose con picazón, estornudos o erupciones.",
        "La gastroenteritis es una inflamación del estómago y el intestino, generalmente viral o bacteriana, que provoca diarrea, dolor abdominal, náuseas y vómitos.",
        "La bronquitis es una inflamación de los bronquios, los conductos que llevan aire a los pulmones, causando tos persistente con o sin flema. Puede ser aguda o crónica.",
        "La neumonía es una infección pulmonar grave que inflama los sacos de aire en uno o ambos pulmones, llenándolos de líquido o pus, lo que dificulta la respiración.",
        "El infarto de miocardio (ataque cardíaco) ocurre cuando el flujo sanguíneo al corazón se bloquea repentinamente, causando daño al músculo cardíaco por falta de oxígeno.",
        "Ante la sospecha de un infarto, llama INMEDIATAMENTE al número de emergencias local (ej. 911 en México). NO conduzcas. Mastica una aspirina si te lo indican los servicios de emergencia (si no eres alérgico). Mantén la calma y espera la ayuda médica. ⚠️ ¡Cada minuto cuenta en esta emergencia médica!",
        "El dengue es una enfermedad viral transmitida por la picadura del mosquito Aedes aegypti, caracterizada por fiebre alta, dolores intensos de cabeza, musculares y articulares, y erupción cutánea. Puede complicarse en casos severos."
    ],
    "Recomendacion": [
        "Para la gripe, se recomienda reposo, hidratación abundante y analgésicos como paracetamol o ibuprofeno para la fiebre y el dolor. Evita el contacto cercano para prevenir contagios. ⚠️ Consulta a un médico si la fiebre persiste más de 3 días, si hay dificultad respiratoria, dolor en el pecho o empeoramiento de los síntomas.",
        "Para Covid-19, se aconseja aislamiento para evitar la propagación. Monitorea tus síntomas y mantente hidratado. Utiliza paracetamol para la fiebre. ⚠️ Busca atención médica inmediata si experimentas dificultad para respirar, dolor persistente en el pecho, confusión nueva o coloración azulada en labios/cara.",
        "Para las alergias, lo principal es evitar el alérgeno desencadenante. Los antihistamínicos orales o nasales pueden aliviar los síntomas. ⚠️ Busca atención médica de urgencia si hay hinchazón de labios, lengua, cara o garganta (angioedema), dificultad respiratoria severa o desmayo, ya que podría ser anafilaxia.",
        "En caso de gastroenteritis, la prioridad es prevenir la deshidratación. Bebe abundantes líquidos como agua, sueros orales o caldos. Consume una dieta blanda (arroz, plátano, tostadas) y evita lácteos o alimentos grasos. ⚠️ Consulta a un médico si la diarrea es persistente (>2 días), si hay fiebre alta, sangre en las heces o signos de deshidratación grave (boca muy seca, orina escasa, letargo).",
        "Para la bronquitis aguda, descansa, bebe muchos líquidos y evita irritantes respiratorios (humo, polvo). Los medicamentos para la tos pueden ser útiles bajo supervisión. ⚠️ Si la tos persiste más de 3 semanas, si hay fiebre alta, escalofríos, dificultad para respirar o expectoración con sangre, consulta a un médico para descartar otras condiciones.",
        "Para la neumonía, es crucial buscar atención médica INMEDIATA. Generalmente requiere hospitalización, administración de antibióticos (si es bacteriana), antivirales (si es viral) y oxígeno. El tratamiento es personalizado según la causa y la gravedad. ⚠️ ¡Es una emergencia médica!",
        "Ante la sospecha de un infarto, llama INMEDIATAMENTE al número de emergencias local (ej. 911 en México). NO conduzcas. Mastica una aspirina si te lo indican los servicios de emergencia (si no eres alérgico). Mantén la calma y espera la ayuda médica. ⚠️ ¡Cada minuto cuenta en esta emergencia médica!",
        "Para el dengue, se recomienda reposo absoluto y una hidratación intensa con agua o sueros orales. Controla la fiebre con paracetamol (evita el ibuprofeno y la aspirina por riesgo de sangrado). Monitorea los signos de alarma. ⚠️ Busca atención médica urgente si presentas dolor abdominal intenso, vómitos persistentes, sangrado (encías, nariz), debilidad extrema o dificultad para respirar."
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
        # Ajustamos el umbral a 60 para que sea un poco menos estricto
        # y permita que más síntomas se consideren "reconocidos" por la IA para el diagnóstico interno.
        mejor, score = process.extractOne(s, mlb.classes_)
        if score >= 60: # Umbral ajustado
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
    except Exception as e: # Captura la excepción para un mejor manejo en logs
        print(f"Error en verificar_tendencia_google: {e}")
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
    except Exception as e: # Captura la excepción para un mejor manejo en logs
        print(f"Error en detectar_brotes_ubicacion: {e}")
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
    # NOTA: sintomas_usuario es la cadena de entrada del formulario.
    # La corregimos para que la IA la entienda mejor, pero el output seguirá usando esta versión corregida para mostrar.
    # Si quieres que se muestre la entrada original, necesitarías una variable separada para eso.
    sintomas_procesados_para_ia = corregir_sintomas(sintomas_usuario) 
    sintomas_numericos = mlb.transform([sintomas_procesados_para_ia])

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
            # Esta línea mostrará los síntomas que fueron "reconocidos" por la función corregir_sintomas.
            # Si fuzzywuzzy con score >= 60 no encuentra coincidencias, esta lista puede estar vacía.
            f"📝 Síntomas reconocidos: {', '.join(sintomas_procesados_para_ia)}\n" 
            f"{alertas_brotes}"
            f"📌 Enfermedad probable: {enfermedad_predicha}\n"
            f"ℹ {enfermedad_info['Descripcion']}\n"
            f"💊 Recomendación: {enfermedad_info['Recomendacion']}\n"
            f"{estado_peso}\n"
            f"{tendencia_google}\n"
            f"{emergencia}"
        )
    except Exception as e: # Captura la excepción para verla en logs
        print(f"Error inesperado en diagnosticar: {e}") # Imprime el error en los logs
        return f"⚠ {nombre}, no se encontró una coincidencia exacta. Consulta a un médico. (Detalle: {e})"

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    # NOTA: La variable 'sintomas_disponibles' NO se pasa a render_template en la solicitud GET inicial
    # en esta versión de tu código, lo cual puede causar errores si el HTML espera esa variable.
    # Si quieres que los síntomas del dropdown se actualicen automáticamente desde app.py,
    # el HTML y esta función necesitarían ser modificados para pasar esa variable siempre.
    
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

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)

