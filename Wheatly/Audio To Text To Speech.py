# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3
import ollama

# Initialize the recognizer
r = sr.Recognizer()

#Models
# hf.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF:Q4_0 = Llama 3
# hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:latest = SmolLM2
# hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0     = TinyLlama

MODEL = 'hf.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF:Q4_0'  #Llama3


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


stream = ollama.chat(
    model=MODEL,
    messages=[{'role': 'user',
               'content': 'Your name is turret'}],
    stream=True
)
answer = ''
for chunk in stream:
    temp_answer = chunk['message']['content']
    answer = answer + temp_answer
print('Active!')

# Loop infinitely for user to
# speak

while True:

    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for half a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.5)

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            if 'controller' in MyText:
                print('You>', MyText)
                stream = ollama.chat(
                    model=MODEL,
                    messages=[{'role': 'user', 'content': MyText}],
                    stream=True
                )
                answer = ''
                for chunk in stream:
                    temp_answer = chunk['message']['content']
                    answer = answer + temp_answer
                print(answer)
                pyttsx3.speak(answer)

    except sr.RequestError as e:
        print('Shits fucked')

    except sr.UnknownValueError:
        pass
