"""Check response delivery while the audio producer is still open."""
import asyncio

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.proto import ux_vad_pb2 as pb
from fireredvad_ncnn.servicer.servicer import UxVoiceActivityDetectorService


def test_default_window_delivers_before_reading_third_chunk():
    received_first_response = False
    consumed = []
    closed = []

    class Engine:
        config = EngineConfig()

        async def open_stream(self, *args):
            pass

        async def stream_chunk(self, session, sequence, audio, sr, final):
            return [(-1.0, -1.0, sequence / 10)]

        async def close_stream(self, *args):
            closed.append(args)

    async def requests():
        yield pb.StreamingDetectRequest(config=pb.DetectionConfig(encoding=1, sample_rate_hertz=16000))
        for i in range(3):
            if i == 2:
                assert received_first_response, "Service read a third chunk before releasing the first response"
            consumed.append(i)
            yield pb.StreamingDetectRequest(audio_content=b'\0\0' * 1600)

    async def run():
        nonlocal received_first_response
        service = UxVoiceActivityDetectorService(Engine())
        responses = service.StreamingDetect(requests(), object())
        first = await anext(responses)
        assert consumed == [0, 1]
        assert first.processed_time.ToNanoseconds() == 100_000_000
        received_first_response = True
        rest = [response async for response in responses]
        assert [r.processed_time.ToNanoseconds() for r in rest] == [200_000_000, 300_000_000, 400_000_000]
        assert len(closed) == 1

    asyncio.run(run())
