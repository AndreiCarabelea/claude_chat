import pyaudio
import wave
import time
import math
import keyboard
import sys


def record_audio(filename):
    """Record audio from microphone and save to file"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        # Record until 'q' is pressed
        while not keyboard.is_pressed('q'):
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except IOError as e:
                print(f"Warning: {e}")
                continue

        print("* done recording")

        # Stop and close the stream properly
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    except Exception as e:
        print(f"Error: {e}")
        # Ensure resources are cleaned up even if an error occurs
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
        if 'stream' in locals():
            stream.close()
        p.terminate()
        raise

    return True

# Usage example
if __name__ == "__main__":
    record_audio("recording.wav")