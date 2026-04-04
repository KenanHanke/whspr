import av


def audio_to_m4a(input_path: str, output_path: str) -> None:
    """Convert an audio file to M4A format with 16kHz mono using PyAV."""
    input_container = av.open(input_path)
    output_container = av.open(output_path, "w")

    try:
        input_stream = next(s for s in input_container.streams if s.type == "audio")

        output_stream = output_container.add_stream("aac", rate=16000)
        output_stream.layout = "mono"
        output_stream.bit_rate = 32000

        resampler = av.audio.resampler.AudioResampler( # type: ignore
            format="fltp",
            layout="mono",
            rate=16000,
        )

        for frame in input_container.decode(input_stream):
            for frame in resampler.resample(frame):
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)

        for frame in resampler.resample(None):
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        for packet in output_stream.encode(None):
            output_container.mux(packet)

    finally:
        output_container.close()
        input_container.close()
