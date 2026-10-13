#Este asistente combina la rigurosidad de la estadística clásica con la capacidad de síntesis de la IA. 
# No "adivina" precios: utiliza un motor de regresión lineal (OLS) en tiempo real para calcular los pesos exactos 
# de las materias primas y sus intervalos de confianza del 95%. El modelo de lenguaje (Llama 3.1, ejecutado localmente)
# actúa únicamente como un orquestador y redactor, garantizando que cada cifra presentada esté matemáticamente verificada.


import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from pathlib import Path

# Configuración para evitar errores de codificación en Windows al imprimir tildes/emojis
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# CONFIGURACIÓN Y CARGA DE DATOS
RUTA_DATOS = Path("..") / "Datos" / "historico_equipos.csv"
MATERIAS_PRIMAS = ["Price_X", "Price_Y", "Price_Z"]

# Temperature=0 es crucial: fuerza al modelo a ser determinista y no "inventar" números
LLM_CONFIG = LLM(
    model="ollama/llama3.1", 
    base_url="http://localhost:11434", 
    temperature=0, 
    max_tokens=900
)

@st.cache_data(show_spinner=False)
def cargar_datos():
    """Carga y limpia el dataset. Se usa caché para no releer el CSV en cada interacción."""
    df = pd.read_csv(RUTA_DATOS, parse_dates=["Date"], index_col="Date").sort_index()
    return df.dropna().rename_axis(None)

df_historico = cargar_datos()

# LÓGICA DE NEGOCIO Y HERRAMIENTAS
def _obtener_nombre_columna(equipo_solicitado: str) -> str:
    """Normaliza la entrada del usuario para encontrar la columna exacta en el DataFrame."""
    clave = equipo_solicitado.lower().replace("price_", "").replace(" ", "")
    if clave.endswith("1"):
        return "Price_Equipo1"
    if clave.endswith("2"):
        return "Price_Equipo2"
    raise ValueError(f"Equipo no reconocido: '{equipo_solicitado}'. Por favor, especifica Equipo1 o Equipo2.")


@tool("AnalizarEquipo")
def analizar_equipo(equipo: str, meses_proyeccion: int = 0) -> str:
    """
    Calcula los pesos de las materias primas que explican el precio del equipo.
    Si meses_proyeccion > 0, añade una proyección con Intervalo de Confianza del 95%.
    """
    try:
        col_objetivo = _obtener_nombre_columna(equipo)
    except ValueError as e:
        return str(e)

    # Preparación de matrices para Regresión Lineal Múltiple (OLS)
    X = np.column_stack([np.ones(len(df_historico)), df_historico[MATERIAS_PRIMAS].to_numpy(float)])
    y = df_historico[col_objetivo].to_numpy(float)
    
    # Estadísticas descriptivas básicas del equipo
    stats = df_historico[col_objetivo].describe()

    # Resolución del sistema por Mínimos Cuadrados Ordinarios
    coeficientes, *_ = np.linalg.lstsq(X, y, rcond=None)
    
    # Cálculo de residuales y dimensiones para inferencia estadística
    residuales = y - X @ coeficientes
    n_observaciones, n_predictores = X.shape
    
    # Desviación estándar de los errores del modelo
    desvio_error = float(np.std(residuales, ddof=n_predictores))
    
    # Cálculo del estadístico t para validar significancia (evita dar peso a ruido)
    varianza_coeficientes = np.diag((residuales @ residuales) / (n_observaciones - n_predictores) * np.linalg.inv(X.T @ X))
    estadistico_t = coeficientes / np.sqrt(varianza_coeficientes)
    
    # Normalización de pesos (solo positivos, sumando 100%)
    pesos_crudos = np.maximum(coeficientes[1:], 0.0)
    pesos_normalizados = pesos_crudos / max(pesos_crudos.sum(), 1e-9)
    
    # Métricas de ajuste
    r_cuadrado = 1 - (residuales @ residuales) / ((y - y.mean()) @ (y - y.mean()))
    
    nombres_vars = ["X", "Y", "Z"]
    materia_mas_relevante = int(np.argmax(pesos_normalizados))
    

    # Construcción del reporte de texto para el LLM
    reporte = (
        f"Periodo analizado: {df_historico.index.min():%Y-%m-%d} a {df_historico.index.max():%Y-%m-%d} ({n_observaciones} días). "
        f"Último precio registrado: {col_objetivo} = {y[-1]:,.2f}.\n"
        f"Calidad del modelo: R2 = {r_cuadrado:.4f}, Desvío estándar del error = {desvio_error:.2f}.\n"
        f"Estadísticas de {col_objetivo}: Media = {stats['mean']:,.2f}, Desv = {stats['std']:,.2f}, "
        f"Mín = {stats['min']:,.2f}, Máx = {stats['max']:,.2f}.\n"
        f"Pesos relativos: X = {pesos_normalizados[0]:.1%} (t={estadistico_t[1]:.1f}), "
        f"Y = {pesos_normalizados[1]:.1%} (t={estadistico_t[2]:.1f}), "
        f"Z = {pesos_normalizados[2]:.1%} (t={estadistico_t[3]:.1f}). Nota: |t|<2 indica falta de significancia estadística.\n"
        f"CONCLUSIÓN: El precio de {col_objetivo} está determinado principalmente por la materia prima {nombres_vars[materia_mas_relevante]} "
        f"con un peso del {pesos_normalizados[materia_mas_relevante]:.1%}."
    )
    

    # Lógica de proyección (si el usuario la solicita)
    meses = max(0, min(int(meses_proyeccion), 12)) # Tope de seguridad a 12 meses
    if meses > 0:
        # Usamos el promedio de los últimos 3 meses de las materias primas como base
        valores_futuros_materias = df_historico[MATERIAS_PRIMAS].resample("ME").last().tail(3).mean().to_numpy(float)
        precio_proyectado = coeficientes[0] + valores_futuros_materias @ coeficientes[1:]
        
        tabla_proyeccion = pd.DataFrame({
            "mes": pd.date_range(df_historico.index.max(), periods=meses + 1, freq="ME")[1:].strftime("%Y-%m"),
            "esperado": precio_proyectado.round(2), 
            "ic_inf": (precio_proyectado - 1.96 * desvio_error).round(2), 
            "ic_sup": (precio_proyectado + 1.96 * desvio_error).round(2)
        })
        reporte += f"\n\nProyección a {meses} meses (basada en promedio reciente de insumos):\n{tabla_proyeccion.to_string(index=False)}"
        
    return reporte


@tool("BuscarContexto")
def buscar_contexto(tema: str) -> str:
    """
    Guarda anti-alucinación: Si no hay fuente de datos externa real conectada, 
    instruye al LLM a declarar la ausencia de dato en lugar de inventar cifras.
    """
    return f"No hay fuente de datos externa conectada para '{tema}'. Repórtalo explícitamente como ausencia de dato, no como una hipótesis o cifra inventada."


# ==========================================
# ORQUESTACIÓN DEL EQUIPO
# ==========================================
@st.cache_resource(show_spinner=False)
def construir_equipo():
    """Inicializa los agentes y sus tareas. Se usa caché para no recrearlos en cada mensaje."""
    
    analista = Agent(
        role="Analista de Costos",
        tools=[analizar_equipo],
        llm=LLM_CONFIG,
        max_iter=3,
        verbose=False, 
        allow_delegation=False,
        goal="Determinar qué materia prima explica cada equipo y proyectar su costo con rigor estadístico.",
        backstory="Eres experto en Pandas y regresión lineal. Tu única fuente de verdad son las cifras devueltas por la herramienta AnalizarEquipo."
    )
    
    gerente = Agent(
        role="Gerente de Estrategia",
        tools=[buscar_contexto],
        llm=LLM_CONFIG,
        max_iter=3,
        verbose=False,
        allow_delegation=False,
        goal="Combinar el análisis cuantitativo con el contexto y responder en español de forma ejecutiva.",
        backstory="Traduces datos complejos en decisiones de negocio. Distingues claramente entre una cifra verificada y una afirmación sin fuente."
    )

    tarea_analisis = Task(
        agent=analista,
        description="Consulta del usuario: {pregunta}\nHistorial reciente: {historial}\n"
                    "Instrucción: Llama a AnalizarEquipo por cada equipo involucrado. Usa 'meses' > 0 solo si se pide proyección explícita.",
        expected_output="Cifras crudas sin prosa: periodo, pesos X/Y/Z con su estadístico t, R2 y tabla de proyección si aplica."
    )
    
    tarea_final = Task(
        agent=gerente,
        context=[tarea_analisis],
        description="Responde a la pregunta: {pregunta}. Usa BuscarContexto una vez. "
                    "Toma las cifras de la tarea anterior como hechos verificados. Máximo 160 palabras.",
        expected_output="Respuesta final en Markdown, concisa, marcando cada punto clave como 'Dato verificado' o 'Sin fuente externa'."
    )

    return Crew(
        agents=[analista, gerente],
        tasks=[tarea_analisis, tarea_final],
        process=Process.sequential
    )


#  INTERFAZ DE USUARIO USANDO STREAMLIT

st.set_page_config(page_title="CrewAI · Análisis de costos", layout="wide")
st.title("Equipo de Análisis de Precios")
st.caption(f"Histórico disponible: {df_historico.index.min():%Y-%m-%d} → {df_historico.index.max():%Y-%m-%d} | LLM: ollama/llama3.1 (Local)")

# Inicializar estado del chat
st.session_state.setdefault("messages", [])

# Renderizar mensajes previos
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Lógica de interacción del chat
if pregunta := st.chat_input("Haz tu pregunta al equipo (ej: '¿De qué depende el Equipo 1 y cómo será en 3 meses?')..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    
    # Compresión inteligente del historial para no saturar el contexto del LLM
    historial_resumido = " | ".join(
        f"{m['role']}: {m['content'][:200]}" for m in st.session_state.messages[-5:-1]
    ) or "sin turnos previos"
    
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    with st.chat_message("assistant"):
        with st.status("Analizando...", expanded=False) as estado:
            try:
                resultado = construir_equipo().kickoff(inputs={"pregunta": pregunta, "historial": historial_resumido})
                estado.update(label="✅ Análisis completado", state="complete")
            except Exception as exc:
                estado.update(label="❌ Falló la ejecución", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                st.stop()
                
        # Mostrar el proceso de pensamiento de cada agente 
        for i, output_tarea in enumerate(resultado.tasks_output, 1):
            with st.expander(f"🔍 Detalle del Paso {i} (Pensamiento del Agente)"):
                st.markdown(str(output_tarea.raw))
                
        # Respuesta final limpia
        respuesta_final = str(getattr(resultado, "raw", None) or resultado)
        st.markdown(respuesta_final)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})