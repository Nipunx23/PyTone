import wave
import speech_recognition as sr
import pyttsx3


# Function to convert text to speech
# def SpeakText(command):
#     engine = pyttsx3.init()
#     engine.say(command)
#     engine.runAndWait()


r = sr.Recognizer()

bb = True

while (bb):

    try:

        with sr.Microphone() as source2:

            # wait for a second to let the recognizer adjust the energy threshold based on the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)
            print("LISTENING...")

            # listens for the user's input
            audio2 = r.listen(source2)
            print("PROCESSING...")

            with wave.open("input_audio.wav", "wb") as wf:
                wf.setnchannels(1)  # Mono audio
                wf.setsampwidth(2)  # 2 bytes per sample
                wf.setframerate(44100)  # Sample rate, you can adjust this
                wf.writeframes(audio2.frame_data)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            with open('input.txt', 'w') as file:
               file.write(MyText)

            print("Did you say ", MyText)
            #SpeakText(MyText)
            try:
                bb = int(input("Try again..?\n"))
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                break

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")





# import pyaudio
# import wave
# import speech_recognition as sr

# frames_per_buffer = 3200
# format = pyaudio.paInt16
# channels = 1
# rate = 44100

# p = pyaudio.PyAudio()
# stream = p.open(format=format, 
#                channels=channels, 
#                rate=rate, 
#                input=True,
#                frames_per_buffer=frames_per_buffer
#               )

# print("LISTENING...")
# seconds = 5
# frames = []

# for i in range(0, int(rate / frames_per_buffer * seconds)):
#     data = stream.read(frames_per_buffer)
#     frames.append(data)

# stream.stop_stream()
# stream.close()
# p.terminate()

# obj = wave.open("input_audio.wav", "wb")
# obj.setnchannels(channels)
# obj.setsampwidth(p.get_sample_size(format))
# obj.setframerate(rate)
# obj.writeframes(b"".join(frames))
# obj.close()

# r = sr.Recognizer()
# audio2 = sr.AudioFile("input_audio.wav")
# print("PROCESSING AUDIO...")

# try:
#     with audio2 as source:
#         audio = r.record(source)  #
#     MyText = r.recognize_google(audio)  
#     MyText = MyText.lower()
#     with open('input.txt', 'w') as file:
#         file.write(MyText)
#     print("Recognized text:", MyText)
# except sr.UnknownValueError:
#     print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print(f"Could not request results from Google Speech Recognition service; {e}")
