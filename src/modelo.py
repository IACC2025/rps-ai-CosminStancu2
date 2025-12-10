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
import random
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
    df = df.copy()

    # --- Frecuencias acumuladas ---
    df['j2_freq_piedra'] = (df['jugada_j2_num'] == 0).expanding().mean()
    df['j2_freq_papel'] = (df['jugada_j2_num'] == 1).expanding().mean()
    df['j2_freq_tijera'] = (df['jugada_j2_num'] == 2).expanding().mean()

    # --- Lags de las últimas jugadas ---
    for lag in range(1, 4):
        df[f'j2_lag_{lag}'] = df['jugada_j2_num'].shift(lag)
    for lag in range(1, 3):
        df[f'j1_lag_{lag}'] = df['jugada_j1_num'].shift(lag)

    # --- Resultado anterior ---
    df['resultado_ronda'] = df.apply(lambda row: 0 if row['jugada_j1']==row['jugada_j2']
                                     else 1 if GANA_A[row['jugada_j1']]==row['jugada_j2'] else 2, axis=1)
    df['resultado_anterior'] = df['resultado_ronda'].shift(1)
    df['gano_j2_anterior'] = (df['resultado_anterior'] == 2).astype(int)
    df['perdio_j2_anterior'] = (df['resultado_anterior'] == 1).astype(int)
    df['empate_anterior'] = (df['resultado_anterior'] == 0).astype(int)

    # --- Patrones y estadísticas avanzadas ---
    df['tijeras_ultimas_10'] = (df['jugada_j2_num'] == 2).rolling(10, min_periods=1).sum()

    # Distancia desde la última tijera
    df['rondas_sin_tijera'] = 0
    ultima = -1
    for idx in range(len(df)):
        if df.loc[idx, 'jugada_j2_num'] == 2:
            ultima = idx
            df.loc[idx, 'rondas_sin_tijera'] = 0
        else:
            df.loc[idx, 'rondas_sin_tijera'] = idx - ultima if ultima >= 0 else idx + 1

    # Ratio tijera/piedra
    df['ratio_tijera_piedra'] = df['jugada_j2_num'].eq(2).cumsum() / (df['jugada_j2_num'].eq(0).cumsum() + 1)

    # Perdió 2 seguidas
    df['perdio_2_seguidas'] = 0
    for idx in range(2, len(df)):
        if df.loc[idx-1,'resultado_ronda']==1 and df.loc[idx-2,'resultado_ronda']==1:
            df.loc[idx,'perdio_2_seguidas'] = 1

    # Detectar patrones del jugador
    df['j1_alterna'] = 0
    df['j1_repite'] = 0
    df['j1_patron_contador'] = 0
    contador = 0
    for idx in range(3, len(df)):
        ultimos = df.loc[idx-3:idx-1, 'jugada_j1_num'].tolist()
        if len(set(ultimos)) == 2 and ultimos[0]==ultimos[2]:
            df.loc[idx,'j1_alterna'] = 1
        if df.loc[idx,'jugada_j1_num'] == df.loc[idx-1,'jugada_j1_num']:
            df.loc[idx,'j1_repite'] = 1
        if df.loc[idx,'j1_alterna']==1 or df.loc[idx,'j1_repite']==1:
            contador += 1
        else:
            contador = 0
        df.loc[idx,'j1_patron_contador'] = contador

    return df

# Lista oficial de features que se usarán SIEMPRE
FEATURES_OFICIALES = [
    'j2_freq_piedra', 'j2_freq_papel', 'j2_freq_tijera',
    'j2_lag_1', 'j2_lag_2', 'j2_lag_3', 'j1_lag_1', 'j1_lag_2',
    'resultado_anterior', 'gano_j2_anterior', 'perdio_j2_anterior', 'empate_anterior',
    'tijeras_ultimas_10', 'rondas_sin_tijera', 'ratio_tijera_piedra', 'perdio_2_seguidas',
    'j1_alterna', 'j1_repite', 'j1_patron_contador'
]

def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    """
    feature_cols = [
        col for col in df.columns
        if col not in [
            'numero_ronda',
            'jugada_j1',
            'jugada_j2',
            'jugada_j1_num',
            'jugada_j2_num',
            'proxima_jugada_j2',
            'resultado',  # ← TEXTO (eliminar)
            'resultado_anterior',  # ← TEXTO (eliminar)
            'resultado_ronda'
        ]
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
        'KNN_3': KNeighborsClassifier(n_neighbors=3),
        'KNN_7': KNeighborsClassifier(n_neighbors=7),
        'Tree_Deep': DecisionTreeClassifier(max_depth=15, min_samples_split=3, random_state=42),
        'RF_200': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        ),
        'RF_300': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42
        )
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
        #hola

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
    mejor_modelo.feature_names = FEATURES_OFICIALES.copy()

    from sklearn.model_selection import cross_val_score

    # Validación cruzada del mejor modelo
    scores = cross_val_score(mejor_modelo, X, y, cv=5, scoring='accuracy')
    print(f"\n📊 Validación Cruzada (5-fold):")
    print(f"   Scores: {scores}")
    print(f"   Media: {scores.mean():.4f} (+/- {scores.std():.4f})")

    if scores.mean() < 0.35:
        print("   ⚠️  WARNING: Accuracy muy baja, el modelo puede no generalizar bien")

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

        # ============================================================
        # FEATURES ANTI-EXPLOTACIÓN (LAS QUE FALTABAN)
        # ============================================================

        # j1_alterna: Detecta si J1 alterna entre 2 jugadas (A-B-A)
        if len(df_hist) >= 3:
            ultimas_3 = df_hist['jugada_j1_num'].tail(3).tolist()
            # Alterna si hay exactamente 2 valores únicos y el primero == tercero
            if len(set(ultimas_3)) == 2 and ultimas_3[0] == ultimas_3[2]:
                features['j1_alterna'] = 1
            else:
                features['j1_alterna'] = 0
        else:
            features['j1_alterna'] = 0

        # j1_repite: Detecta si J1 repite la misma jugada que la anterior
        if len(df_hist) >= 2:
            features['j1_repite'] = 1 if df_hist['jugada_j1_num'].iloc[-1] == df_hist['jugada_j1_num'].iloc[-2] else 0
        else:
            features['j1_repite'] = 0

        # j1_patron_contador: Cuenta cuántas rondas consecutivas J1 ha seguido un patrón
        features['j1_patron_contador'] = 0
        if len(df_hist) >= 2:
            contador = 0
            for i in range(1, len(df_hist)):
                # Verificar si repite
                repite = df_hist['jugada_j1_num'].iloc[i] == df_hist['jugada_j1_num'].iloc[i - 1]

                # Verificar si alterna (necesita al menos 3 valores)
                alterna = False
                if i >= 2:
                    ultimas = [
                        df_hist['jugada_j1_num'].iloc[i - 2],
                        df_hist['jugada_j1_num'].iloc[i - 1],
                        df_hist['jugada_j1_num'].iloc[i]
                    ]
                    if len(set(ultimas)) == 2 and ultimas[0] == ultimas[2]:
                        alterna = True

                # Si sigue algún patrón, incrementar contador
                if repite or alterna:
                    contador += 1
                else:
                    contador = 0

            features['j1_patron_contador'] = contador

        # ============================================================
        # CONVERTIR A ARRAY EN EL MISMO ORDEN QUE ENTRENAMIENTO
        # ============================================================
        feature_order = self.modelo.feature_names if self.modelo is not None else FEATURES_OFICIALES

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

    def decidir_jugada(self):
        """
        IA combinada: predicción del modelo, detección de patrones recientes,
        contrarresto directo y aleatoriedad adaptativa.
        """
        PIERDE_CONTRA = {
            "piedra": "papel",
            "papel": "tijera",
            "tijera": "piedra"
        }

        # 1. Obtener features actuales
        f = self.obtener_features_actuales()

        # Si no hay features → contrarresto directo o aleatorio
        if f is None:
            if len(self.historial) > 0:
                ultima_hum = self.historial[-1][0]
                return PIERDE_CONTRA[ultima_hum]
            return np.random.choice(["piedra", "papel", "tijera"])

        # 2. Predicción del modelo
        try:
            proba = self.modelo.predict_proba([f])[0]  # Probabilidades
            pred_num = np.argmax(proba)
            pred_hum = NUM_A_JUGADA[pred_num]
            confianza = proba[pred_num]  # Nivel de certeza
        except:
            pred_hum = np.random.choice(["piedra", "papel", "tijera"])
            confianza = 0.33

        # 3. Aleatoriedad adaptativa según confianza del modelo
        if np.random.random() < (1 - confianza) or np.random.random() < 0.20:
            return np.random.choice(["piedra", "papel", "tijera"])

        # 4. Jugada que le gana a la predicción del humano
        jugada_modelo = PIERDE_CONTRA[pred_hum]

        # 5. Evitar empates con la última jugada del humano
        if len(self.historial) >= 1:
            ultima_hum = self.historial[-1][0]
            if jugada_modelo == ultima_hum:
                jugada_modelo = np.random.choice([x for x in ["piedra", "papel", "tijera"] if x != ultima_hum])

        # 6. Detectar empates recientes para romper bucle
        if len(self.historial) >= 2:
            e1 = self.historial[-1][0] == self.historial[-1][1]
            e2 = self.historial[-2][0] == self.historial[-2][1]
            if e1 and e2:
                jugada_modelo = PIERDE_CONTRA[self.historial[-1][0]]

        # 7. Ajustes por patrones recientes del humano
        if len(self.historial) >= 3:
            ult_3_hum = [h for (h, _) in self.historial[-3:]]
            ult_3_ia = [ia for (_, ia) in self.historial[-3:]]
            # Romper patrón si humano copia IA
            if ult_3_hum[1] == ult_3_ia[0] and ult_3_hum[2] == ult_3_ia[1]:
                jugada_modelo = np.random.choice(["piedra", "papel", "tijera"])
            # Aprovechar repetición del humano
            elif ult_3_hum[-1] == ult_3_hum[-2]:
                jugada_modelo = PIERDE_CONTRA[ult_3_hum[-1]]

        return jugada_modelo

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
    df = cargar_datos("D:/PCS/rps-ai-CosminStancu2/data/Datos_Keko_Ñete_Final_Cut.csv")  #C:/Users/Usuario/PycharmProjects/rps-ai-CosminStancu2/data/Datos_Keko_Ñete_Final_Cut.csv   #

    # 2. Preparar datos
    print("\n[2/5] Preparando datos...")
    df = preparar_datos(df)

    # 3. Crear features
    print("\n[3/5] Creando features...")
    df = crear_features(df)

    #cuando yo la vi jujujuju
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