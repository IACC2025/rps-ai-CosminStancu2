"""
RPSAI - Evaluador de Winrate (MEJORADO)
========================================

Este script evalua el rendimiento de tu modelo de IA
jugando partidas contra un humano y calculando el winrate.

Uso:
    python src/evaluador.py
    python src/evaluador.py -n 100  # Para jugar 100 rondas
    python src/evaluador.py --modelo ruta/al/modelo.pkl  # Ruta personalizada

El evaluador:
1. Carga tu modelo entrenado
2. Juega N partidas contra un humano
3. Calcula y muestra el winrate final
4. Muestra la nota segun los criterios de evaluacion
"""

import sys

import os

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Agregar el directorio src al path para importar modelo
sys.path.insert(0, str(Path(__file__).parent))

from modelo import JugadorIA, JUGADA_A_NUM, NUM_A_JUGADA, GANA_A, RUTA_MODELO


# Mapeo de entrada a jugada
ENTRADA_A_JUGADA = {
    "1": "piedra", "2": "papel", "3": "tijera",
    "p": "piedra", "a": "papel", "t": "tijera",
    "piedra": "piedra", "papel": "papel", "tijera": "tijera"
}


def obtener_nota(winrate: float) -> tuple:
    """
    Calcula la nota basada en el winrate.

    Returns:
        (nota, mensaje_especial)
    """
    if winrate >= 55:
        return 10, "🎉 BONUS: Fallos en examen no restan!"
    elif winrate >= 50:
        return 10, None
    elif winrate >= 49:
        return 9, None
    elif winrate >= 48:
        return 8, None
    elif winrate >= 46:
        return 7, None
    elif winrate >= 44:
        return 6, None
    elif winrate >= 42:
        return 5, None
    elif winrate >= 40:
        return 4, None
    elif winrate >= 39:
        return 3, None
    elif winrate >= 37:
        return 2, None
    elif winrate >= 35:
        return 1, None
    else:
        return 0, None


def obtener_resultado(jugada_ia: str, jugada_humano: str) -> str:
    """Obtiene el resultado desde la perspectiva de la IA."""
    if jugada_ia == jugada_humano:
        return "empate"
    elif GANA_A[jugada_ia] == jugada_humano:
        return "victoria"
    else:
        return "derrota"


def leer_jugada_humano() -> str:
    """Lee la jugada del humano."""
    while True:
        print("\n[1/p] Piedra  [2/a] Papel  [3/t] Tijera  [q] Salir")
        entrada = input("Tu jugada: ").lower().strip()

        if entrada == 'q':
            print("\n❌ Evaluación cancelada por el usuario.")
            sys.exit(0)

        if entrada in ENTRADA_A_JUGADA:
            return ENTRADA_A_JUGADA[entrada]

        print("❌ Jugada no valida. Intenta de nuevo.")


def mostrar_ronda(ronda: int, jugada_ia: str, jugada_humano: str, resultado: str):
    """Muestra el resultado de una ronda."""
    simbolos = {"piedra": "🪨", "papel": "📄", "tijera": "✂️"}

    print(f"\n{'='*50}")
    print(f"   RONDA {ronda}")
    print(f"{'='*50}")
    print(f"  Tú:  {simbolos[jugada_humano]}  {jugada_humano.upper()}")
    print(f"  IA:  {simbolos[jugada_ia]}  {jugada_ia.upper()}")
    print(f"{'='*50}")

    if resultado == "victoria":
        print("  🤖 >>> IA GANA <<<")
    elif resultado == "derrota":
        print("  👤 >>> TÚ GANAS <<<")
    else:
        print("  🤝 >>> EMPATE <<<")
    print(f"{'='*50}")


def mostrar_progreso(victorias: int, derrotas: int, empates: int, total: int):
    """Muestra el progreso actual."""
    jugadas = victorias + derrotas + empates
    restantes = total - jugadas

    if victorias + derrotas > 0:
        winrate_actual = victorias / (victorias + derrotas) * 100
        print(f"\n📊 Progreso: {jugadas}/{total} rondas")
        print(f"   IA: {victorias}V - {derrotas}D - {empates}E")
        print(f"   Winrate actual: {winrate_actual:.1f}%")
        print(f"   Rondas restantes: {restantes}")


def cargar_modelo_ia(ruta_modelo: str = None) -> JugadorIA:
    """
    Carga el modelo de IA de forma robusta.

    Args:
        ruta_modelo: Ruta personalizada al modelo (opcional)

    Returns:
        Instancia de JugadorIA con modelo cargado
    """
    print("\n" + "="*60)
    print("   🤖 CARGANDO MODELO DE IA")
    print("="*60)

    # Determinar ruta del modelo
    if ruta_modelo is None:
        ruta_modelo = RUTA_MODELO

    print(f"\n📂 Buscando modelo en: {ruta_modelo}")

    # Verificar si el archivo existe
    if not os.path.exists(ruta_modelo):
        print(f"\n❌ ERROR: No se encontró el modelo en {ruta_modelo}")
        print("\n💡 Soluciones:")
        print("   1. Entrena el modelo primero: python src/modelo.py")
        print("   2. Verifica que el archivo modelo_entrenado.pkl existe en la carpeta 'models/'")
        print("   3. Especifica una ruta personalizada: --modelo ruta/al/modelo.pkl")
        print("\n⚠️  La IA jugará ALEATORIAMENTE sin modelo entrenado.")

        respuesta = input("\n¿Continuar con IA aleatoria? [s/N]: ").lower().strip()
        if respuesta != 's':
            print("❌ Evaluación cancelada.")
            sys.exit(1)

        print("\n⚠️  Continuando con IA ALEATORIA...")
        ia = JugadorIA()  # Modelo None, jugará aleatorio
        return ia

    # Intentar cargar el modelo
    try:
        print(f"\n⏳ Cargando modelo...")
        ia = JugadorIA(ruta_modelo=str(ruta_modelo))

        if ia.modelo is None:
            print("⚠️  ADVERTENCIA: El modelo se cargó pero está vacío (None)")
            print("⚠️  La IA jugará de forma ALEATORIA")
        else:
            print("✅ Modelo cargado exitosamente!")
            print(f"   Tipo de modelo: {type(ia.modelo).__name__}")

            # Mostrar info adicional si está disponible
            if hasattr(ia.modelo, 'n_features_in_'):
                print(f"   Features esperadas: {ia.modelo.n_features_in_}")

        return ia

    except Exception as e:
        print(f"\n❌ ERROR al cargar el modelo: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        print("\n💡 Posibles causas:")
        print("   - El archivo está corrupto")
        print("   - Versión incompatible de sklearn/pickle")
        print("   - El modelo no se guardó correctamente")
        print("\n⚠️  La IA jugará de forma ALEATORIA")

        respuesta = input("\n¿Continuar con IA aleatoria? [s/N]: ").lower().strip()
        if respuesta != 's':
            print("❌ Evaluación cancelada.")
            sys.exit(1)

        ia = JugadorIA()
        ia.modelo = None  # Asegurar que sea None
        return ia


def evaluar(num_rondas: int = 50, ruta_modelo: str = None):
    """
    Ejecuta la evaluacion del modelo.

    Args:
        num_rondas: Numero de rondas a jugar
        ruta_modelo: Ruta personalizada al modelo
    """
    print("\n" + "="*60)
    print("   🎮 RPSAI - EVALUACION DE WINRATE")
    print("="*60)
    print(f"\n📋 Se jugarán {num_rondas} rondas contra tu modelo de IA.")
    print("   Juega de forma natural, como lo harías normalmente.")

    # Cargar el modelo de IA
    ia = cargar_modelo_ia(ruta_modelo)

    # Verificar estado del modelo
    if ia.modelo is None:
        print("\n⚠️  MODO ALEATORIO ACTIVADO")
        print("   La IA no tiene modelo entrenado y jugará aleatoriamente.")
        print("   Los resultados NO serán representativos del rendimiento real.")
    else:
        print("\n✅ MODELO CARGADO CORRECTAMENTE")
        print("   La IA usará el modelo entrenado para predecir tus jugadas.")

    input("\n▶️  Presiona ENTER para comenzar la evaluación...")

    # Contadores
    victorias = 0
    derrotas = 0
    empates = 0

    # Jugar rondas
    for ronda in range(1, num_rondas + 1):
        # La IA decide su jugada (predice la tuya y juega lo que le gana)
        jugada_ia = ia.decidir_jugada()

        # El humano juega
        jugada_humano = leer_jugada_humano()

        # Determinar resultado (desde perspectiva IA)
        resultado = obtener_resultado(jugada_ia, jugada_humano)

        # Mostrar resultado
        mostrar_ronda(ronda, jugada_ia, jugada_humano, resultado)

        # Registrar en el historial de la IA
        # IMPORTANTE: El orden es (tu_jugada, jugada_ia) para que el modelo aprenda
        ia.registrar_ronda(jugada_ia, jugada_humano)

        # Actualizar contadores
        if resultado == "victoria":
            victorias += 1
        elif resultado == "derrota":
            derrotas += 1
        else:
            empates += 1

        # Mostrar progreso
        mostrar_progreso(victorias, derrotas, empates, num_rondas)

    # Resultados finales
    print("\n" + "="*60)
    print("   📊 RESULTADOS FINALES")
    print("="*60)

    total_decisivas = victorias + derrotas
    if total_decisivas > 0:
        winrate = victorias / total_decisivas * 100
    else:
        winrate = 0

    print(f"\n🎮 Rondas jugadas: {num_rondas}")
    print(f"🤖 Victorias IA: {victorias}")
    print(f"👤 Derrotas IA: {derrotas}")
    print(f"🤝 Empates: {empates}")
    print(f"\n{'='*60}")
    print(f"   🏆 WINRATE DE LA IA: {winrate:.1f}%")
    print(f"{'='*60}")

    nota, bonus = obtener_nota(winrate)
    print(f"\n{'='*60}")
    print(f"   📝 NOTA MODO 1 (Winrate): {nota}/10")
    if bonus:
        print(f"   {bonus}")
    print(f"{'='*60}")

    # Interpretación del resultado
    print("\n💡 Interpretación:")
    if ia.modelo is None:
        print("   ⚠️  Resultados con IA ALEATORIA (no representativos)")
        print("   ℹ️  Entrena el modelo para obtener resultados reales")
    elif winrate >= 50:
        print("   🎉 ¡Excelente! Tu modelo supera el 50% de winrate")
        print("   ✅ El modelo ha aprendido patrones de tus jugadas")
    elif winrate >= 40:
        print("   👍 Bien! El modelo tiene un rendimiento decente")
        print("   💡 Puedes mejorar añadiendo más features o datos")
    else:
        print("   ⚠️  El modelo necesita mejoras")
        print("   💡 Revisa las features o entrena con más datos")

    # Tabla de referencia
    print("\n📋 Tabla de referencia de notas:")
    print("  ├─ 33%- = 0  |  35% = 1  |  37% = 2  |  39% = 3")
    print("  ├─ 40% = 4   |  42% = 5  |  44% = 6  |  46% = 7")
    print("  ├─ 48% = 8   |  49% = 9  |  50%+ = 10")
    print("  └─ 55%+ = 10 + BONUS (fallos no restan)")


def main():
    """Funcion principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evalúa el winrate de tu modelo de IA en Piedra-Papel-Tijera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python src/evaluador.py                    # 50 rondas con modelo por defecto
  python src/evaluador.py -n 100             # 100 rondas
  python src/evaluador.py --modelo mi_modelo.pkl  # Modelo personalizado
        """
    )

    parser.add_argument(
        "-n", "--rondas",
        type=int,
        default=50,
        help="Número de rondas a jugar (default: 50)"
    )

    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        help="Ruta personalizada al modelo entrenado (default: models/modelo_entrenado.pkl)"
    )
    args = parser.parse_args()

    # Validar número de rondas
    if args.rondas < 1:
        print("❌ Error: El número de rondas debe ser al menos 1")
        sys.exit(1)

    if args.rondas > 1000:
        respuesta = input(f"⚠️  {args.rondas} rondas es mucho. ¿Continuar? [s/N]: ")
        if respuesta.lower().strip() != 's':
            print("❌ Evaluación cancelada.")
            sys.exit(0)

    # Ejecutar evaluación
    try:
        evaluar(args.rondas, args.modelo)
    except KeyboardInterrupt:
        print("\n\n❌ Evaluación interrumpida por el usuario (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()