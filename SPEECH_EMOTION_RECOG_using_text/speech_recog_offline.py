import speech_recognition as sr
import wave

# Initialize the recognizer
r = sr.Recognizer()

bb = True

while (bb):

    try:
       with sr.Microphone() as source2:
            # Wait for a second to let the recognizer adjust the energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("LISTENING...")

            # Listens for the user's input
            audio2 = r.listen(source2)
            print("PROCESSING...")

            with wave.open("input_audio.wav", "wb") as wf:
                wf.setnchannels(1)  # Mono audio
                wf.setsampwidth(2)  # 2 bytes per sample
                wf.setframerate(44100)  # Sample rate, you can adjust this
                wf.writeframes(audio2.frame_data)

            # Using CMU Sphinx to recognize audio offline
            try:
                MyText = r.recognize_sphinx(audio2)
                MyText = MyText.lower()
                with open('input.txt', 'w') as file:
                    file.write(MyText)
                print("Did you say:", MyText)
            except sr.UnknownValueError:
                print("Sphinx could not understand audio")
            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))

            try:
                bb = int(input("Try again..?\n"))
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                break

    except sr.WaitTimeoutError:
        print("No speech input detected. Please try again.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

print("Goodbye")
