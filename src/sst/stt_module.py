import speech_recognition as sr
from transformers import pipeline

def initialize_stt_pipeline(model_name="openai/whisper-small"):
    """
    Inicializa el pipeline de Speech-to-Text de Transformers (Whisper).

    Args:
        model_name (str): Nombre del modelo Whisper a usar (ej: 'openai/whisper-base', 'openai/whisper-large-v2').

    Returns:
        transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline: El pipeline STT inicializado.
    """
    try:
        print(f"Cargando modelo STT: {model_name}...")
        stt_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        print(f"Modelo STT {model_name} cargado correctamente.")
        return stt_pipeline
    except Exception as e:
        print(f"Error al cargar el modelo STT {model_name}: {e}")
        return None

def speech_to_text(stt_pipeline):
    """
    Captura audio del micrófono y lo convierte a texto usando el pipeline STT.
    """
    import speech_recognition as sr
    r = sr.Recognizer()
    r.pause_threshold = 2.0 # Tiempo de silencio para considerar el final de una frase
    print("speech_to_text: Recognizer inicializado.") # DEBUG
    try:
        with sr.Microphone(device_index=2) as source:
            print("speech_to_text: Micrófono abierto.") # DEBUG
            print(f"speech_to_text: Sample rate del micrófono: {source.SAMPLE_RATE}") # DEBUG
            print(f"speech_to_text: Sample width del micrófono: {source.SAMPLE_WIDTH}") # DEBUG
            print("Habla...")
            try:
                print("speech_to_text: Iniciando r.listen...") # DEBUG
                audio = r.listen(source, phrase_time_limit=120) # Escucha hasta 120 segundos
                print("speech_to_text: Audio capturado.") # DEBUG
                wav_data = audio.get_wav_data() # Obtiene los datos en formato WAV
                print("speech_to_text: Datos WAV obtenidos.") # DEBUG
                audio_data = sr.AudioData(wav_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH) # Crea un objeto AudioData explícitamente como WAV
                print("speech_to_text: Objeto AudioData creado.") # DEBUG
                texto = stt_pipeline(audio_data.get_raw_data())["text"] # Usa audio_data
                print(f"Texto reconocido: {texto}")
                return texto
            except sr.WaitTimeoutError:
                print("No se detectó audio en el tiempo límite.")
                return None
            except Exception as e:
                print(f"speech_to_text: Error al capturar audio del micrófono (dentro del listen): {e}") # DEBUG
                return None
    except Exception as e:
        print(f"speech_to_text: Error al inicializar el micrófono: {e}") # DEBUG
        return None