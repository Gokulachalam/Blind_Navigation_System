import speech_recognition as sr
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)

# Initialize speech recognition engine
r = sr.Recognizer()

# Define function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define function to listen for voice commands
def listen():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        speak("Please speak your command")
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio)
        return command
    except:
        speak("Sorry, I did not understand your command.")
        return ""

# Define function to give directions
def give_directions():
    speak("Please turn left in 10 feet")
    time.sleep(5)
    speak("Please continue straight for 20 feet")
    time.sleep(5)
    speak("Please turn right in 5 feet")

# Main program loop
while True:
    command = listen()

    if "start navigation" in command.lower():
        speak("Starting navigation")
        give_directions()

    elif "stop navigation" in command.lower():
        speak("Stopping navigation")
        break

    else:
        speak("Please say start navigation or stop navigation.")
