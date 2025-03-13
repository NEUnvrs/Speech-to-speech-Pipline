import speech_recognition as sr
from transformers import pipeline
from src.sst import stt_module  # Importa el módulo stt_module
from src.llm import llm_module  # ¡Importa el módulo llm_module!
from src.tts import tts_module  # ¡Importa el módulo tts_module de Kokoro!

def main():
    """
    Función principal para ejecutar el programa Speech-to-Speech con Kokoro TTS.
    Inicializa los módulos STT, LLM y TTS, y luego entra en un bucle
    para procesar la entrada de voz del usuario y obtener respuestas habladas del LLM con Kokoro TTS.
    """
    print("Inicializando programa Speech-to-Speech con Kokoro TTS...")

    # --- Inicialización del módulo STT ---
    stt_pipeline = stt_module.initialize_stt_pipeline()
    if not stt_pipeline:
        print("Error al inicializar el pipeline STT. El programa no puede continuar.")
        return
    print("Módulo STT inicializado correctamente.")

    # --- Inicialización del módulo LLM ---
    llm_initialized = llm_module.initialize_llm_pipeline()
    if not llm_initialized:
        print("Error al inicializar el modelo LLM. El programa no puede continuar.")
        return
    print("Módulo LLM inicializado correctamente.")

    # --- Inicialización del módulo TTS Kokoro ---
    tts_pipeline_kokoro = tts_module.initialize_tts_engine() # Inicializa el motor TTS Kokoro
    if not tts_pipeline_kokoro:
        print("Error al inicializar el motor TTS Kokoro. El programa no puede continuar.")
        return
    print("Módulo TTS Kokoro inicializado correctamente.")

    print("\n--- Programa Speech-to-Speech con Kokoro TTS Listo ---")
    print("¡Ahora puedes empezar a hablar!")

    while True: # Bucle principal para conversación continua
        print("\nHabla...")
        texto_input = stt_module.speech_to_text(stt_pipeline)

        if texto_input:
            print(f"Texto del usuario (STT): {texto_input}")

            # --- Obtener respuesta del LLM ---
            print("Procesando texto con el modelo LLM...")
            respuesta_llm = llm_module.get_llm_response(texto_input)

            if respuesta_llm:
                print(f"Respuesta del LLM: {respuesta_llm}")
                # --- ¡Llama al módulo TTS Kokoro para hablar la respuesta del LLM! ---
                tts_module.text_to_speech(tts_pipeline_kokoro, respuesta_llm) # Usa TTS Kokoro para decir la respuesta del LLM
            else:
                print("Error al obtener respuesta del LLM.")
        else:
            print("No se recibió texto desde el módulo STT. ¿Has hablado?")

    print("\nPrograma Speech-to-Speech con Kokoro TTS - Finalizado.")

if __name__ == "__main__":
    main()