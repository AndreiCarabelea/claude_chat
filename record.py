import pyaudio
import wave
import threading

class AudioRecorder:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=2, rate=44100):
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self._p = None
        self._stream = None
        self._frames = []
        self.is_recording = False
        self._thread = None

    def start(self):
        if self.is_recording:
            print("Already recording.")
            return
        
        self._frames = []
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(format=self.FORMAT,
                                     channels=self.CHANNELS,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.CHUNK)
        self.is_recording = True
        
        self._thread = threading.Thread(target=self._record)
        self._thread.start()
        print("* recording started")

    def _record(self):
        while self.is_recording:
            try:
                data = self._stream.read(self.CHUNK, exception_on_overflow=False)
                self._frames.append(data)
            except IOError:
                # Can happen if stream is closed while reading
                pass

    def stop(self, filename="output.wav"):
        if not self.is_recording:
            print("Not recording.")
            return

        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        if self._p:
            self._p.terminate()
        
        print("* done recording")

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self._p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        
        return True