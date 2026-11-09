import asyncio
import pytest
from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.engine.client import EngineClient
from fireredvad_ncnn.engine.errors import FailedPreconditionError, ResourceExhaustedError
from tests.test_engine_client import factory  # Shared native boundary fixture.


def test_session_capacity_and_duplicate_ids(factory):
    async def run():
        client=EngineClient('unused',EngineConfig(max_sessions=1))
        await client.start()
        try:
            await client.open_stream('s',False)
            with pytest.raises(FailedPreconditionError): await client.open_stream('s',False)
            with pytest.raises(ResourceExhaustedError): await client.open_stream('other',False)
            await client.close_stream('s')
            await client.open_stream('other',False)
        finally: await client.close()
    asyncio.run(run())
