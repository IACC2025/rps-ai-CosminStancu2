"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import data

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    TODO: Implementa esta funcion
    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

    df = pd.read_csv(ruta_csv)

    columnas_necesarias = ['Nº Ronda', 'Cosmin', 'Keko Ñete']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {columnas_faltantes}")

    df = df.rename(columns = {
        'Nº Ronda': 'numero_ronda',
        'Cosmin': 'jugada_j1',
        'Keko Ñete': 'jugada_j2'
    })

    print(f"✅ Datos cargados: {len(df)} rondas")
    print(f"📋 Columnas: {list(df.columns)}")

    return df

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    TODO: Implementa esta funcion
    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    """
    df = df.copy()

    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    df = df.dropna(subset=['proxima_jugada_j2'])

    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    print(f"✅ Datos preparados: {len(df)} rondas válidas")
    print(f"📊 Columnas numéricas creadas: jugada_j1_num, jugada_j2_num, proxima_jugada_j2")

    return df

# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    TODO: Implementa al menos 3 tipos de features diferentes.

    Ideas de features:
    1. Frecuencia de cada jugada del oponente (j2)
    2. Ultimas N jugadas (lag features)
    3. Resultado de la ronda anterior
    4. Racha actual (victorias/derrotas consecutivas)
    5. Patron despues de ganar/perder
    6. Fase del juego (inicio/medio/final)

    Cuantas mas features relevantes crees, mejor podra predecir tu modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    df = df.copy() 

    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()

    df['j2_freq_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean()
    df['j2_freq_papel'] = (df['jugada_j2_num'] == 1).expanding().mean()
    df['j2_freq_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean()

    print("Todo ok")

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.

    df['j2_lag_1'] = df['jugada_j2_num'].shift(1)
    df['j2_lag_2'] = df['jugada_j2_num'].shift(2)
    df['j2_lag_3'] = df['jugada_j2_num'].shift(3)

    #jugadas del jugador

    df['j1_lag_1'] = df['jugada_j1_num'].shift(1)
    df['j1_lag_2'] = df['jugada_j1_num'].shift(2)

    print("Todo ok")

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)

    def calcular_resultado(row):
        """Determina quién ganó la ronda"""
        j1 = row['jugada_j1']
        j2 = row['jugada_j2']

        if j1 == j2:
            return 0  # Empate
        elif GANA_A[j1] == j2:
            return 1  # Gana J1
        else:
            return 2  # Gana J2

    df['resultado_ronda'] = df.apply(calcular_resultado, axis=1)
    df['resultado_anterior'] = df['resultado_ronda'].shift(1)

    # Flags binarios para facilitar al modelo
    df['gano_j2_anterior'] = (df['resultado_anterior'] == 2).astype(int)
    df['perdio_j2_anterior'] = (df['resultado_anterior'] == 1).astype(int)
    df['empate_anterior'] = (df['resultado_anterior'] == 0).astype(int)

    print("Todo ok")
    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Feature 1: CONTADOR DE TIJERAS RECIENTES (últimas 10 rondas)
    df['tijeras_ultimas_10'] = (df['jugada_j2_num'] == 2).rolling(window=10, min_periods=1).sum()

    # Feature 2: DISTANCIA DESDE ÚLTIMA TIJERA
    # Cuántas rondas han pasado desde que jugó tijera
    df['rondas_sin_tijera'] = 0
    ultima_tijera = -1
    for idx in range(len(df)):
        if df.loc[idx, 'jugada_j2_num'] == 2:  # Es tijera
            ultima_tijera = idx
            df.loc[idx, 'rondas_sin_tijera'] = 0
        else:
            df.loc[idx, 'rondas_sin_tijera'] = idx - ultima_tijera if ultima_tijera >= 0 else idx + 1

    # Feature 3: RATIO TIJERA/PIEDRA ACUMULADO
    # Captura el desequilibrio entre sus jugadas favoritas
    # Keko juega tijera (43.5%) casi el doble que piedra (25%)
    piedras_acum = (df['jugada_j2_num'] == 0).cumsum()
    tijeras_acum = (df['jugada_j2_num'] == 2).cumsum()
    # Evitar división por cero
    df['ratio_tijera_piedra'] = tijeras_acum / (piedras_acum + 1)  # +1 para evitar división por 0

    # Feature 4: FLAG "ACABA DE PERDER 2+ VECES"
    # Momento crítico donde Keko tiende a cambiar agresivamente
    # Flag perdio 2 seguidas
    df['perdio_2_seguidas'] = 0
    for idx in range(2, len(df)):
        if df.loc[idx - 1, 'resultado_ronda'] == 1 and df.loc[idx - 2, 'resultado_ronda'] == 1:
            df.loc[idx, 'perdio_2_seguidas'] = 1

    # ---- SIEMPRE SE CALCULAN ESTAS ----
    # Alterna
    df['j1_alterna'] = 0
    for idx in range(3, len(df)):
        j = [
            df.loc[idx - 3, 'jugada_j1_num'],
            df.loc[idx - 2, 'jugada_j1_num'],
            df.loc[idx - 1, 'jugada_j1_num']
        ]
        if len(set(j)) == 2 and j[0] == j[2]:
            df.loc[idx, 'j1_alterna'] = 1

    # Repite
    df['j1_repite'] = (df['jugada_j1_num'] == df['j1_lag_1']).astype(int)

    # Contador de patrón
    df['j1_patron_contador'] = 0
    contador = 0
    for idx in range(1, len(df)):
        if df.loc[idx, 'j1_repite'] == 1 or df.loc[idx, 'j1_alterna'] == 1:
            contador += 1
        else:
            contador = 0
        df.loc[idx, 'j1_patron_contador'] = contador

        print("   ✅ Feature Anti-Explotación añadida")

    print("   ✅ Features específicas Keko Ñete: tijeras recientes, distancia tijera, ratio, flag derrotas")
    print("   ✅ Features adicionales: Cambio de jugada y tendencias")

    # ==========================================
    # RESUMEN
    # ==========================================
    print(
        f"\n📊 Total de features creadas: {len([col for col in df.columns if col not in ['numero_ronda', 'jugada_j1', 'jugada_j2', 'jugada_j1_num', 'jugada_j2_num', 'proxima_jugada_j2']])}")

    return df



def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    """
    # Definir las columnas que usaremos como features
    feature_cols = [
        # Frecuencias
        'j2_freq_piedra',
        'j2_freq_papel',
        'j2_freq_tijera',

        # Lags (jugadas anteriores)
        'j2_lag_1',
        'j2_lag_2',
        'j2_lag_3',
        'j1_lag_1',
        'j1_lag_2',

        # Resultado anterior
        'resultado_anterior',
        'gano_j2_anterior',
        'perdio_j2_anterior',
        'empate_anterior',

        # Features específicas de Keko
        'tijeras_ultimas_10',
        'rondas_sin_tijera',
        'ratio_tijera_piedra',
        'perdio_2_seguidas',

        # Anti-explotación
        'j1_alterna',
        'j1_repite',
        'j1_patron_contador'

    ]

    # Verificar que todas las columnas existen
    columnas_faltantes = [col for col in feature_cols if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"❌ Faltan columnas: {columnas_faltantes}")

    # Crear X (features) e y (target)
    X = df[feature_cols].copy()
    y = df['proxima_jugada_j2'].copy()

    # Eliminar filas con valores nulos
    filas_antes = len(X)
    mask_validos = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask_validos]
    y = y[mask_validos]
    filas_despues = len(X)

    print(f"\n✅ Features seleccionadas")
    print(f"   📊 {len(feature_cols)} features")
    print(f"   📈 {filas_despues} muestras válidas (eliminadas {filas_antes - filas_despues} con NaN)")
    print(f"   🎯 Target: proxima_jugada_j2")

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.
    """
    print("\n" + "=" * 70)
    print("🤖 ENTRENANDO MODELOS")
    print("=" * 70)

    # Dividir los datos en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y  # Mantener proporción de clases
    )

    print(f"\n📊 Datos divididos:")
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Test: {len(X_test)} muestras")

    # Definir modelos a probar
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }

    mejor_modelo = None
    mejor_accuracy = 0
    mejor_nombre = ""

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        print(f"\n{'=' * 70}")
        print(f"📈 Entrenando: {nombre}")
        print(f"{'=' * 70}")

        # Entrenar
        modelo.fit(X_train, y_train)

        # Predecir
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        # Calcular accuracy
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f"\n   Accuracy Train: {acc_train:.4f} ({acc_train * 100:.2f}%)")
        print(f"   Accuracy Test:  {acc_test:.4f} ({acc_test * 100:.2f}%)")

        # Mostrar classification report
        print(f"\n   📋 Classification Report (Test):")
        target_names = ['piedra', 'papel', 'tijera']
        print(classification_report(y_test, y_pred_test, target_names=target_names, digits=3))

        # Mostrar matriz de confusión
        print(f"   📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"        Pred: Piedra  Papel  Tijera")
        for i, label in enumerate(target_names):
            print(f"   Real {label:7s}: {cm[i]}")

        # Guardar el mejor modelo
        if acc_test > mejor_accuracy:
            mejor_accuracy = acc_test
            mejor_modelo = modelo
            mejor_nombre = nombre

    # Resumen final
    print("\n" + "=" * 70)
    print("🏆 MEJOR MODELO")
    print("=" * 70)
    print(f"   Modelo: {mejor_nombre}")
    print(f"   Accuracy Test: {mejor_accuracy:.4f} ({mejor_accuracy * 100:.2f}%)")
    print("=" * 70)

    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # Cargar el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("✅ Modelo cargado correctamente")
        except FileNotFoundError:
            print("⚠️ Modelo no encontrado. Entrena primero con main()")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.
        Deben ser LAS MISMAS features que usaste para entrenar.
        """
        if len(self.historial) < 3:
            # No hay suficiente historial, retornar features vacías
            return None

        # Convertir historial a DataFrame para calcular features
        df_hist = pd.DataFrame(self.historial, columns=['jugada_j1', 'jugada_j2'])
        df_hist['jugada_j1_num'] = df_hist['jugada_j1'].map(JUGADA_A_NUM)
        df_hist['jugada_j2_num'] = df_hist['jugada_j2'].map(JUGADA_A_NUM)

        # Calcular features (DEBEN SER LAS MISMAS que en entrenamiento)
        features = {}

        # Frecuencias
        total = len(df_hist)
        features['j2_freq_piedra'] = (df_hist['jugada_j2_num'] == 0).sum() / total
        features['j2_freq_papel'] = (df_hist['jugada_j2_num'] == 1).sum() / total
        features['j2_freq_tijera'] = (df_hist['jugada_j2_num'] == 2).sum() / total

        # Lags (últimas jugadas)
        features['j2_lag_1'] = df_hist['jugada_j2_num'].iloc[-1] if len(df_hist) >= 1 else 0
        features['j2_lag_2'] = df_hist['jugada_j2_num'].iloc[-2] if len(df_hist) >= 2 else 0
        features['j2_lag_3'] = df_hist['jugada_j2_num'].iloc[-3] if len(df_hist) >= 3 else 0
        features['j1_lag_1'] = df_hist['jugada_j1_num'].iloc[-1] if len(df_hist) >= 1 else 0
        features['j1_lag_2'] = df_hist['jugada_j1_num'].iloc[-2] if len(df_hist) >= 2 else 0

        # Resultado anterior
        if len(df_hist) >= 1:
            j1_ant = df_hist['jugada_j1'].iloc[-1]
            j2_ant = df_hist['jugada_j2'].iloc[-1]

            if j1_ant == j2_ant:
                resultado = 0  # Empate
            elif GANA_A[j1_ant] == j2_ant:
                resultado = 1  # Gana J1
            else:
                resultado = 2  # Gana J2

            features['resultado_anterior'] = resultado
            features['gano_j2_anterior'] = 1 if resultado == 2 else 0
            features['perdio_j2_anterior'] = 1 if resultado == 1 else 0
            features['empate_anterior'] = 1 if resultado == 0 else 0
        else:
            features['resultado_anterior'] = 0
            features['gano_j2_anterior'] = 0
            features['perdio_j2_anterior'] = 0
            features['empate_anterior'] = 0

        # Features específicas de Keko
        ultimas_10 = df_hist['jugada_j2_num'].tail(10)
        features['tijeras_ultimas_10'] = (ultimas_10 == 2).sum()

        # Distancia desde última tijera
        tijeras_idx = df_hist[df_hist['jugada_j2_num'] == 2].index
        if len(tijeras_idx) > 0:
            features['rondas_sin_tijera'] = len(df_hist) - tijeras_idx[-1] - 1
        else:
            features['rondas_sin_tijera'] = len(df_hist)

        # Ratio tijera/piedra
        piedras = (df_hist['jugada_j2_num'] == 0).sum()
        tijeras = (df_hist['jugada_j2_num'] == 2).sum()
        features['ratio_tijera_piedra'] = tijeras / (piedras + 1)

        # Perdió 2 seguidas
        if len(df_hist) >= 2:
            # Calcular resultados de las últimas 2 rondas
            resultados = []
            for i in range(len(df_hist) - 2, len(df_hist)):
                j1 = df_hist['jugada_j1'].iloc[i]
                j2 = df_hist['jugada_j2'].iloc[i]
                if j1 == j2:
                    res = 0
                elif GANA_A[j1] == j2:
                    res = 1
                else:
                    res = 2
                resultados.append(res)

            features['perdio_2_seguidas'] = 1 if all(r == 1 for r in resultados) else 0
        else:
            features['perdio_2_seguidas'] = 0

        # ==========================================
        # FEATURES ANTI-EXPLOTACIÓN (LAS QUE FALTABAN)
        # ==========================================

        # j1_alterna: Detecta si J1 alterna (ej: piedra-papel-piedra)
        if len(df_hist) >= 3:
            j = [
                df_hist['jugada_j1_num'].iloc[-3],
                df_hist['jugada_j1_num'].iloc[-2],
                df_hist['jugada_j1_num'].iloc[-1]
            ]
            features['j1_alterna'] = 1 if (len(set(j)) == 2 and j[0] == j[2]) else 0
        else:
            features['j1_alterna'] = 0

        # j1_repite: J1 repite la misma jugada que la anterior
        if len(df_hist) >= 2:
            features['j1_repite'] = 1 if df_hist['jugada_j1_num'].iloc[-1] == df_hist['jugada_j1_num'].iloc[-2] else 0
        else:
            features['j1_repite'] = 0

        # j1_patron_contador: Cuenta cuántas rondas seguidas J1 ha seguido un patrón
        features['j1_patron_contador'] = 0
        if len(df_hist) >= 2:
            contador = 0
            for i in range(1, len(df_hist)):
                # Repite
                repite = df_hist['jugada_j1_num'].iloc[i] == df_hist['jugada_j1_num'].iloc[i - 1]

                # Alterna (si hay al menos 3 rondas)
                alterna = False
                if i >= 2:
                    j = [
                        df_hist['jugada_j1_num'].iloc[i - 2],
                        df_hist['jugada_j1_num'].iloc[i - 1],
                        df_hist['jugada_j1_num'].iloc[i]
                    ]
                    alterna = len(set(j)) == 2 and j[0] == j[2]

                if repite or alterna:
                    contador += 1
                else:
                    contador = 0

            features['j1_patron_contador'] = contador

        # Convertir a array en el MISMO ORDEN que entrenamiento
        feature_order = [
            'j2_freq_piedra', 'j2_freq_papel', 'j2_freq_tijera',
            'j2_lag_1', 'j2_lag_2', 'j2_lag_3', 'j1_lag_1', 'j1_lag_2',
            'resultado_anterior', 'gano_j2_anterior', 'perdio_j2_anterior', 'empate_anterior',
            'tijeras_ultimas_10', 'rondas_sin_tijera', 'ratio_tijera_piedra', 'perdio_2_seguidas',
            'j1_alterna', 'j1_repite', 'j1_patron_contador'  # ← LAS 3 QUE FALTABAN
        ]

        return np.array([features[f] for f in feature_order])

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        # Obtener features actuales
        features = self.obtener_features_actuales()

        if features is None:
            # No hay suficiente historial
            return np.random.choice(["piedra", "papel", "tijera"])

        # Predecir con el modelo
        prediccion_num = self.modelo.predict([features])[0]
        prediccion_texto = NUM_A_JUGADA[prediccion_num]

        return prediccion_texto

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.
    """
    print("=" * 50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    df = cargar_datos("C:/Users/Usuario/PycharmProjects/rps-ai-CosminStancu2/data/Datos_Keko_Ñete_Final_Cut.csv") #ruta pc casa: C:/Users/Usuario/PycharmProjects/rps-ai-CosminStancu2/data/Datos_Keko_Ñete_Final_Cut.csv   ruta pc insti:D:/PCS/rps-ai-CosminStancu2/data/Datos_Keko_Ñete_Final_Cut.csv

    # 2. Preparar datos
    print("\n[2/5] Preparando datos...")
    df = preparar_datos(df)

    # 3. Crear features
    print("\n[3/5] Creando features...")
    df = crear_features(df)

    # 4. Seleccionar features
    print("\n[4/5] Seleccionando features...")
    X, y = seleccionar_features(df)

    # 5. Entrenar modelo
    print("\n[5/5] Entrenando modelos...")
    modelo = entrenar_modelo(X, y, test_size=0.2)

    # 6. Guardar modelo
    print("\n💾 Guardando modelo...")
    guardar_modelo(modelo)

    print("\n" + "=" * 50)
    print("✅ ¡ENTRENAMIENTO COMPLETADO!")
    print("=" * 50)
    print("\nAhora puedes usar JugadorIA para jugar:")
    print("  ia = JugadorIA()")
    print("  jugada = ia.decidir_jugada()")

if __name__ == "__main__":
    main()
