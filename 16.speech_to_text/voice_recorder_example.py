import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

device_info = sd.query_devices(None, 'input')
samplerate = int(device_info['default_samplerate'])
filename = "test_voice.wav"
try:
    with sf.SoundFile(filename, mode='w', samplerate=samplerate,
                    channels=1, subtype=None) as file:
        with sd.InputStream(samplerate=samplerate, device=None,
                            channels=1, callback=callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())
except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(filename))
except Exception as e:
    print(e)