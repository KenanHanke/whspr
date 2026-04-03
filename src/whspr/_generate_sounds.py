#!/usr/bin/env python3

import math
import wave
import struct
from typing import List
import os
import random

DEFAULT_SAMPLE_RATE = 44100


def main():
    length = 0.3

    C3_half = synth_guitar_note(130.81, length/2)
    G3_half = synth_guitar_note(196.00, length/2)
    C4_whole = synth_guitar_note(261.63, length)
    F2_whole = synth_guitar_note(87.31, length)

    start_sound = C3_half + G3_half
    end_sound = G3_half + C3_half
    finished_sound = C4_whole
    cancelled_sound = F2_whole

    sounds_dir = os.path.join(os.path.dirname(__file__), "data", "sounds")
    os.makedirs(sounds_dir, exist_ok=True)

    write_wav(os.path.join(sounds_dir, "start.wav"), start_sound)
    write_wav(os.path.join(sounds_dir, "end.wav"), end_sound)
    write_wav(os.path.join(sounds_dir, "finished.wav"), finished_sound)
    write_wav(os.path.join(sounds_dir, "cancelled.wav"), cancelled_sound)

    # Finally, play all the sounds using aplay (Linux), each time
    # prompting the user to press Enter to continue.
    for name in ["start", "end", "finished", "cancelled"]:
        input(f"Press enter to play {name}.wav...")
        os.system(f"aplay {os.path.join(sounds_dir, name + '.wav')}")



def synth_piano_note(
    frequency: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.5,
) -> List[float]:
    """
    Generate a simple piano-like note as a list of floats in [-1.0, 1.0].

    This is intentionally simple, not a physical piano model.
    It uses a few slightly inharmonic partials plus a fast attack
    and exponential decay.
    """
    if frequency <= 0:
        raise ValueError("frequency must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if duration == 0:
        return []

    n_samples = int(duration * sample_rate)

    # A few piano-ish partials:
    # (frequency multiplier, amplitude, decay multiplier)
    partials = [
        (1.00, 1.00, 3.8),
        (2.01, 0.55, 5.2),
        (3.02, 0.28, 6.8),
        (4.08, 0.14, 8.5),
    ]

    attack_time = 0.005  # 5 ms
    samples: List[float] = []

    for i in range(n_samples):
        t = i / sample_rate

        # Fast attack
        if t < attack_time:
            attack = t / attack_time
        else:
            attack = 1.0

        # Main decay envelope
        decay = math.exp(-4.5 * t / duration)

        sample = 0.0
        for mult, amp, decay_mult in partials:
            env = attack * math.exp(-decay_mult * t / duration)
            sample += amp * env * math.sin(2.0 * math.pi * frequency * mult * t)

        # Small hammer/noise-like transient can help, but keeping it minimal:
        # no noise here for simplicity.

        sample *= volume * decay
        samples.append(sample)

    # Normalize lightly if needed
    peak = max(abs(x) for x in samples) or 1.0
    if peak > 1.0:
        samples = [x / peak for x in samples]

    # Simple fade-out at the end to avoid clicks
    fade_samples = int(0.02 * sample_rate)  # 20 ms fade
    fade_samples = min(fade_samples, n_samples//4)  # Don't exceed quarter the note length
    for i in range(fade_samples):
        t = i / fade_samples
        fade_factor = t**2  # Quadratic fade
        samples[-1 - i] *= fade_factor

    return samples

def synth_guitar_note(
    frequency: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    volume: float = 0.5,
) -> List[float]:
    """
    Generate a simple guitar-like note as a list of floats in [-1.0, 1.0].

    This uses a very small Karplus-Strong style plucked-string model.
    """
    if frequency <= 0:
        raise ValueError("frequency must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if duration == 0:
        return []

    n_samples = int(duration * sample_rate)

    # Delay length determines pitch.
    delay = max(2, int(sample_rate / frequency))

    # Initial burst of noise = pluck excitation.
    buffer = [random.uniform(-1.0, 1.0) for _ in range(delay)]

    samples: List[float] = []
    index = 0
    decay = 0.996

    for _ in range(n_samples):
        current = buffer[index]

        i_m2 = (index - 2) % delay
        i_m1 = (index - 1) % delay
        i_p1 = (index + 1) % delay
        i_p2 = (index + 2) % delay

        new_value = (
            0.0625 * buffer[i_m2]
            + 0.25   * buffer[i_m1]
            + 0.375  * buffer[index]
            + 0.25   * buffer[i_p1]
            + 0.0625 * buffer[i_p2]
        ) * decay

        buffer[index] = new_value
        samples.append(current * volume)
        index = i_p1

    # Light normalization if needed
    peak = max(abs(x) for x in samples) or 1.0
    if peak > 1.0:
        samples = [x / peak for x in samples]

    # Simple fade-out at the end to avoid clicks
    fade_samples = int(0.02 * sample_rate)  # 20 ms fade
    fade_samples = min(fade_samples, n_samples//4)  # Don't exceed quarter the note length
    for i in range(fade_samples):
        t = i / fade_samples
        fade_factor = t**2  # Quadratic fade
        samples[-1 - i] *= fade_factor

    return samples

def write_wav(
    filename: str,
    samples: List[float],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """
    Write mono 16-bit PCM WAV using the standard library wave module.
    Expects samples roughly in [-1.0, 1.0].
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    pcm_frames = bytearray()

    for x in samples:
        # Clamp to [-1.0, 1.0]
        x = max(-1.0, min(1.0, x))
        # Convert to signed 16-bit
        pcm = int(x * 32767.0)
        pcm_frames.extend(struct.pack("<h", pcm))

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_frames)

if __name__ == "__main__":
    main()