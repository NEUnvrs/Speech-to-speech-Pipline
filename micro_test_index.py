import speech_recognition as sr

for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone with name \"{microphone_name}\" found")