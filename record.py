import pyaudio
import wave
import time
import math
import keyboard


def record_audio(filename, sample_rate=48000, channels=2, chunk=256, stop_key='q'):
    """
    Record audio and save it as a WAV file.

    Parameters:
    - filename: str, the name of the output WAV file
    - duration: int, recording duration in seconds (default: 5)
    - sample_rate: int, number of samples per second (default: 44100)
    - channels: int, number of audio channels (1 for mono, 2 for stereo) (default: 1)
    - chunk: int, number of frames per buffer (default: 1024)
    """
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"Recording... Press '{stop_key}' to stop.")
    
    frames = []

    while not keyboard.is_pressed(stop_key):
        try:
            data = stream.read(chunk)
            frames.append(data)
        except OSError as e:
            print(f"Error: {e}")
            break
    
        
      # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")

 
    # Save the recorded audio as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved as {filename}")

# Usage example
if __name__ == "__main__":
    record_audio("recording.wav")