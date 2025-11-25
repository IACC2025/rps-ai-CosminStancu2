# RPSAI - Modulo principal
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Datos originales exactos del CSV
data_raw = """Nº Ronda,Cosmin,Keko Ñete
1,t,t
2,r,s
3,t,s
4,s,t
5,t,r
6,s,s
7,r,t
8,r,r
9,r,r
10,s,s
11,r,r
12,t,s
13,t,r
14,s,r
15,r,t
16,s,s
17,t,t
18,t,s
19,s,s
20,r,r
21,r,s
22,r,t
23,s,t
24,r,r
25,r,s
26,s,s
27,t,r
28,t,t
29,r,s
30,r,r
31,s,r
32,r,r
33,s,t
34,t,t
35,s,s
36,r,t
37,s,r
38,r,s
39,r,r
40,r,s
41,s,t
42,s,r
43,t,r
44,s,r
45,t,t
46,s,r
47,t,t
48,r,s
49,s,r
50,s,s
51,t,r
52,s,s
53,r,t
54,s,t
55,r,t
56,s,s
57,r,r
58,s,s
59,r,r
60,t,r
61,t,t
62,t,r
63,t,s
64,t,s
65,s,t
66,t,r
67,r,s
68,t,r
69,s,s
70,r,s
71,t,s
72,r,s
73,t,r
74,t,r
75,s,s
76,r,t
77,s,s
78,t,s
79,r,r
80,s,t
81,r,s
82,s,s
83,r,t
84,t,r
85,t,s
86,s,s
87,r,r
88,s,t
89,r,s
90,t,r
91,s,s
92,t,r
93,s,r
94,t,t
95,t,s
96,r,r
97,t,t
98,r,s
99,r,t
100,s,t
101,s,t
102,t,r
103,s,t
104,r,s
105,t,s
106,s,t
107,r,r
108,r,r
109,s,s
110,r,t
111,t,s
112,t,t
113,t,s
114,t,s
115,s,s
116,t,t
117,r,s
118,r,s
119,t,t
120,s,r
121,r,r
122,r,t
123,t,s
124,s,r
125,r,t
126,t,s
127,r,t
128,s,s
129,t,t
130,s,t
131,t,s
132,t,r
133,t,s
134,r,t
135,s,s
136,r,r
137,s,s
138,t,t
139,s,s
140,r,t
141,s,t
142,s,s
143,s,t
144,s,s
145,t,s
146,s,t
147,r,r
148,s,r
149,r,r
150,s,t
151,r,s
152,s,s
153,t,t
154,s,r
155,r,t
156,t,t
157,s,r
158,t,s
159,s,r
160,t,s
161,s,t
162,s,s
163,r,t
164,s,t
165,r,t
166,s,r
167,t,r
168,t,t
169,t,t
170,s,r
171,t,s
172,s,r
173,t,t
174,s,r
175,r,s
176,s,t
177,r,t
178,s,s
179,t,r
180,s,s
181,t,t
182,s,s
183,r,r
184,t,t
185,s,s
186,t,r
187,s,s
188,r,s
189,s,t
190,r,s
191,s,t
192,s,t
193,r,s
194,s,s
195,r,r
196,s,s
197,t,r
198,s,s
199,t,r
200,s,s"""

# Mapeo de abreviaturas a nombres completos
MAP = {'r': 'piedra', 's': 'tijera', 't': 'papel'}

# Crear DataFrame desde el CSV original
lines = data_raw.strip().split('\n')
header = lines[0].split(',')

data = []
for line in lines[1:]:  # Saltar la primera línea (encabezado)
    parts = line.split(',')
    data.append({
        'Nº Ronda': int(parts[0]),
        'Cosmin': MAP[parts[1]],
        'Keko Ñete': MAP[parts[2]]
    })

df = pd.DataFrame(data)


# ==========================================
# CALCULAR QUIEN GANA CADA RONDA
# ==========================================
def quien_gana(cosmin, keko):
    """Determina quién gana la ronda"""
    if cosmin == keko:
        return 'empate'

    # Reglas del juego
    if (cosmin == 'piedra' and keko == 'tijera') or \
            (cosmin == 'tijera' and keko == 'papel') or \
            (cosmin == 'papel' and keko == 'piedra'):
        return 'gana_cosmin'
    else:
        return 'gana_keko'


df['resultado'] = df.apply(lambda row: quien_gana(row['Cosmin'], row['Keko Ñete']), axis=1)

# ==========================================
# 1. RESULTADO ANTERIOR
# ==========================================
df['resultado_anterior'] = df['resultado'].shift(1)
df.loc[0, 'resultado_anterior'] = 'ninguno'

# ==========================================
# 2. FRECUENCIAS ACUMULADAS DE KEKO
# ==========================================
df['keko_freq_piedra'] = (df['Keko Ñete'] == 'piedra').cumsum() / df['Nº Ronda']
df['keko_freq_papel'] = (df['Keko Ñete'] == 'papel').cumsum() / df['Nº Ronda']
df['keko_freq_tijera'] = (df['Keko Ñete'] == 'tijera').cumsum() / df['Nº Ronda']

# Redondear a 4 decimales
df['keko_freq_piedra'] = df['keko_freq_piedra'].round(4)
df['keko_freq_papel'] = df['keko_freq_papel'].round(4)
df['keko_freq_tijera'] = df['keko_freq_tijera'].round(4)


# ==========================================
# 3. ENTROPÍA DE KEKO (ventana de 10 rondas)
# ==========================================
def calcular_entropia(jugadas, ventana=10):
    """Calcula la entropía en una ventana deslizante"""
    map_num = {'piedra': 0, 'papel': 1, 'tijera': 2}
    jugadas_num = [map_num[j] for j in jugadas]

    entropias = []
    for i in range(len(jugadas_num)):
        if i < ventana - 1:
            # No hay suficientes datos para calcular entropía
            entropias.append(None)
        else:
            # Tomar las últimas 'ventana' jugadas
            window = jugadas_num[i - ventana + 1: i + 1]
            valores, conteos = np.unique(window, return_counts=True)
            probabilidades = conteos / len(window)
            ent = entropy(probabilidades, base=2)
            entropias.append(round(ent, 4))

    return entropias


df['keko_entropia'] = calcular_entropia(df['Keko Ñete'].values, ventana=10)

# ==========================================
# 4. RACHAS DE KEKO
# ==========================================
df['keko_racha_victorias'] = 0
df['keko_racha_derrotas'] = 0
df['keko_racha_empates'] = 0

racha_vic = 0
racha_der = 0
racha_emp = 0

for idx in range(len(df)):
    if idx == 0:
        # Primera ronda: no hay racha previa
        df.loc[idx, 'keko_racha_victorias'] = 0
        df.loc[idx, 'keko_racha_derrotas'] = 0
        df.loc[idx, 'keko_racha_empates'] = 0
    else:
        # Ver el resultado de la ronda ANTERIOR
        resultado_prev = df.loc[idx - 1, 'resultado']

        if resultado_prev == 'gana_keko':
            racha_vic += 1
            racha_der = 0
            racha_emp = 0
        elif resultado_prev == 'gana_cosmin':
            racha_vic = 0
            racha_der += 1
            racha_emp = 0
        else:  # empate
            racha_vic = 0
            racha_der = 0
            racha_emp += 1

        df.loc[idx, 'keko_racha_victorias'] = racha_vic
        df.loc[idx, 'keko_racha_derrotas'] = racha_der
        df.loc[idx, 'keko_racha_empates'] = racha_emp

# ==========================================
# ORDENAR COLUMNAS
# ==========================================
columnas_finales = [
    'Nº Ronda',
    'Cosmin',
    'Keko Ñete',
    'resultado',
    'resultado_anterior',
    'keko_freq_piedra',
    'keko_freq_papel',
    'keko_freq_tijera',
    'keko_entropia',
    'keko_racha_victorias',
    'keko_racha_derrotas',
    'keko_racha_empates'
]

df_final = df[columnas_finales]

# ==========================================
# GUARDAR CSV
# ==========================================
nombre_archivo = 'partidas.csv'
df_final.to_csv(nombre_archivo, index=False)

print("✅ CSV generado correctamente: partidas.csv")
print(f"📊 Total de columnas: {len(df_final.columns)}")
print(f"📈 Total de filas: {len(df_final)}")
print("\n" + "=" * 70)
print("COLUMNAS DEL CSV:")
print("=" * 70)
print("\n📌 ORIGINALES (3):")
print("  1. Nº Ronda - Número de la ronda (1-200)")
print("  2. Cosmin - Jugada de Cosmin (piedra/papel/tijera)")
print("  3. Keko Ñete - Jugada de Keko Ñete (piedra/papel/tijera)")
print("\n📌 FEATURES AÑADIDAS DE KEKO ÑETE (9):")
print("  4. resultado - Quién ganó (gana_keko/gana_cosmin/empate)")
print("  5. resultado_anterior - Resultado de la ronda previa")
print("  6. keko_freq_piedra - Frecuencia acumulada de piedra")
print("  7. keko_freq_papel - Frecuencia acumulada de papel")
print("  8. keko_freq_tijera - Frecuencia acumulada de tijera")
print("  9. keko_entropia - Entropía (predictibilidad) últimas 10 rondas")
print(" 10. keko_racha_victorias - Victorias consecutivas de Keko")
print(" 11. keko_racha_derrotas - Derrotas consecutivas de Keko")
print(" 12. keko_racha_empates - Empates consecutivos")
print("\n" + "=" * 70)
print("PRIMERAS 15 FILAS:")
print("=" * 70)
print(df_final.head(15).to_string(index=False))
print("\n" + "=" * 70)
print("EJEMPLO DE DATOS AVANZADOS (Rondas 50-55):")
print("=" * 70)
print(df_final.iloc[49:55].to_string(index=False))