import os
import subprocess
import wave

import imageio_ffmpeg
import numpy as np


def add_alert_sound(tracked, times, final, sr=44100):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", tracked]
    wav = None
    if times:
        n = int((max(times) + 2) * sr)
        track = np.zeros(n, dtype=np.float32)
        t = np.arange(int(0.15 * sr)) / sr
        beep = (0.6 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        for s in times:
            for k in range(3):
                a = int((s + k * 0.3) * sr)
                track[a:a + len(beep)] += beep
        wav = final + ".wav"
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
        cmd += ["-i", wav, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", final]
    subprocess.run(cmd, check=True, capture_output=True)
    if wav and os.path.exists(wav):
        os.remove(wav)
