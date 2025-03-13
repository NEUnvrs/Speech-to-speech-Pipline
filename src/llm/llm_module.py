import os

def initialize_llm_pipeline():
    # """
    # Inicializa el pipeline del Modelo de Lenguaje (LLM) usando Ollama.

    # En esta versión inicial, la "inicialización" del LLM con Ollama
    # es bastante sencilla.  Simplemente verificamos que Ollama esté
    # disponible y listo para ser usado en la función `get_llm_response`.

    # Retorna:
    #     bool: True si la inicialización del LLM se considera exitosa (en esta etapa),
    #           False si hay algún problema inicial (aunque la mayoría de los errores
    #           se manejarán al intentar obtener una respuesta en `get_llm_response`).
    # """
    try:
        print("Inicializando el modelo LLM DeepSeek Coder 7B Instruct Distill R1 con Ollama...")
        # En esta versión, no hacemos una inicialización compleja aquí.
        # La inicialización real del modelo (carga, etc.) la maneja Ollama
        # cuando lo llamamos en `get_llm_response`.
        print("Modelo LLM DeepSeek Coder 7B Instruct Distill R1 inicializado (a través de Ollama).")
        return True  # Indicamos que la inicialización inicial fue exitosa
    except Exception as e:
        print(f"Error al inicializar el modelo LLM: {e}")
        return False

def get_llm_response(user_prompt):
    """
    Obtiene una respuesta del modelo LLM DeepSeek Coder 7B Instruct Distill R1 usando Ollama.
    (Versión Modificada para extraer la respuesta después de </think>)
    """
    try:
        modelo_llm_ollama = "deepseek-r1:7b" # Nombre del modelo en Ollama
        ollama_command = f'ollama run {modelo_llm_ollama} "{user_prompt}"'

        print(f"Enviando prompt al LLM Ollama: {user_prompt}")

        process = os.popen(ollama_command)
        llm_output = process.read()
        process.close()

        print("Respuesta cruda del LLM Ollama:\n", llm_output) # Imprime la salida cruda para debugging

        respuesta_llm_texto = ""
        inicio_respuesta = False
        for linea in llm_output.splitlines():
            if inicio_respuesta:
                respuesta_llm_texto += linea + "\n"
            if "</think>" in linea: # Busca la etiqueta de cierre </think>
                inicio_respuesta = True

        respuesta_llm_texto = respuesta_llm_texto.strip()

        print(f"Texto de respuesta del LLM extraído (después de </think>):\n", respuesta_llm_texto) # Mensaje modificado

        if not respuesta_llm_texto:
            print("Advertencia: No se pudo extraer texto de respuesta del LLM desde la salida de Ollama (después de </think>).") # Mensaje modificado
            return None

        return respuesta_llm_texto

    except Exception as e:
        print(f"Error al obtener respuesta del LLM Ollama: {e}")
        return None

    except Exception as e:
        print(f"Error al obtener respuesta del LLM Ollama: {e}")
        return None

if __name__ == '__main__':
    # --- Pruebas del módulo LLM ---
    print("--- Pruebas del módulo llm_module ---")

    if initialize_llm_pipeline():
        print("\nInicialización del LLM exitosa.")

        prompt_de_prueba = "¿Cuál es la capital de ESPAÑA?"
        respuesta = get_llm_response(prompt_de_prueba)

        if respuesta:
            print(f"\nPrompt de prueba: {prompt_de_prueba}")
            print(f"Respuesta del LLM:\n{respuesta}")
        else:
            print("\nError: No se pudo obtener respuesta del LLM para el prompt de prueba.")
    else:
        print("\nError: Falló la inicialización del LLM.")

    print("\n--- Pruebas del módulo llm_module finalizadas ---")