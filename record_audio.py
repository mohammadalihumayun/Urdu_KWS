import pyaudio
import wave
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
ww = wave.open('xx/s11111001.wav')
CHANNELS =  ww.getnchannels()
RATE = ww.getframerate()
RECORD_SECONDS = 0.5
ww.close()
WAVE_OUTPUT_FILENAME = "xx/testrecord.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

input("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


#os.remove('xx/testrecord.wav')
