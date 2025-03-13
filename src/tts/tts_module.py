from kokoro import KPipeline
# from IPython.display import Audio # ...
import soundfile as sf
import torch
import numpy as np

def initialize_tts_engine(lang_code='a', voice='af_heart'): # 'a' para American English, 'af_heart' es una voz predeterminada
    # """
    # Inicializa el pipeline de Text-to-Speech (TTS) Kokoro.

    # Args:
    #     lang_code (str): Código de idioma para Kokoro (ej: 'a' para inglés americano, 'j' para japonés).
    #     voice (str o torch.Tensor): Nombre de la voz predeterminada de Kokoro (ej: 'af_heart')
    #                                  o un tensor de voz cargado.

    # Retorna:
    #     kokoro.KPipeline: El pipeline TTS de Kokoro inicializado, o None si hay un error.
    # """
    try:
        print(f"Inicializando motor TTS Kokoro con idioma '{lang_code}' y voz '{voice}'...")
        pipeline = KPipeline(lang_code=lang_code) # Inicializa el pipeline Kokoro con el código de idioma
        print(f"Motor TTS Kokoro inicializado correctamente con idioma '{lang_code}' y voz '{voice}'.")
        return pipeline
    except Exception as e:
        print(f"Error al inicializar el motor TTS Kokoro: {e}")
        return None

def text_to_speech(pipeline, texto, voice='af_heart'): # Eliminamos output_filename
    """
    Convierte texto a voz usando Kokoro TTS, acumulando audio y permitiendo interrupción con 'Esc'.
    """
    import sounddevice as sd
    import keyboard  # Importa la biblioteca keyboard para detectar teclas
    import time     # Importa time para pausas cortas

    try:
        print(f"TTS Kokoro: Convirtiendo texto a voz (con interrupción 'Esc'): '{texto}'")
        generator = pipeline(
            texto, voice=voice,
            speed=1, split_pattern=r'\n+'
        )

        sample_rate = 24000
        audio_completo = []

        for i, (gs, ps, audio_chunk) in enumerate(generator):
            audio_np = audio_chunk.cpu().numpy()
            audio_completo.extend(audio_np)

        audio_completo_np = np.array(audio_completo)

        print("TTS Kokoro: Reproduciendo audio... (Presiona 'Esc' para interrumpir)")
        sd.play(audio_completo_np, samplerate=sample_rate)

        while sd.get_stream().active: # Mientras el stream de audio esté activo (reproduciendo)
            if keyboard.is_pressed('esc'): # Comprueba si se ha pulsado la tecla 'Esc'
                print("TTS Kokoro: Interrupción por usuario (tecla 'Esc'). Deteniendo reproducción.")
                sd.stop() # Detiene la reproducción de audio inmediatamente
                break # Sale del bucle while (y termina la función text_to_speech)
            time.sleep(0.1) # Pausa breve para no consumir CPU innecesariamente en el bucle de comprobación de tecla

        sd.wait() # Asegura que la reproducción se detenga completamente (por si acaso)
        print("TTS Kokoro: Reproducción finalizada (o interrumpida).")

    except Exception as e:
        print(f"Error en TTS Kokoro: {e}")


if __name__ == '__main__':
    # --- Pruebas del módulo TTS Kokoro ---
    print("--- Pruebas del módulo tts_module Kokoro ---")

    tts_pipeline = initialize_tts_engine() # Inicializa el motor TTS Kokoro
    if tts_pipeline:
        print("\nInicialización del TTS Kokoro exitosa.")

        texto_de_prueba = "Hola, este es un mensaje de prueba del módulo Text-to-Speech Kokoro."
        print(f"\nTexto de prueba para TTS Kokoro: '{texto_de_prueba}'")
        text_to_speech(tts_pipeline, texto_de_prueba, "kokoro_test_output.wav") # Llama a la función TTS Kokoro y guarda el audio en "kokoro_test_output.wav"

        print("\nPrueba de TTS Kokoro completada. Audio guardado en 'kokoro_test_output.wav'.")
    else:
        print("\nError: Falló la inicialización del TTS Kokoro.")

    print("\n--- Pruebas del módulo tts_module Kokoro finalizadas ---")