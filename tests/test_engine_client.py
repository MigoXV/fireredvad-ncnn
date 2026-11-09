"""Async admission, cancellation, ownership and shutdown at the native boundary."""
import asyncio
import threading
import time
from collections import deque

import numpy as np
import pytest

import fireredvad_ncnn.engine.client as module
from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.engine.client import EngineClient
from fireredvad_ncnn.engine.errors import (
    FailedPreconditionError, ResourceExhaustedError, UnavailableError, InvalidArgumentError,
)
from fireredvad_ncnn.native import Completion, NativeError


class FakeNative:
    def __init__(self, *args, **kwargs):
        self.owner = threading.get_ident()
        self.calls = []
        self.sessions = set()
        self.requests = {}
        self.results = deque()
        self.canceled = set()
        self.gate = threading.Event()
        self.gate.set()
        self.open_gate = threading.Event()
        self.open_gate.set()
        self.open_entered = threading.Event()
        self.closed = False
        self.fail = False
        self.build_info = 'fake'
        self.next_id = 0
        self.events = [(0., 1., 1.)]

    def check(self):
        assert threading.get_ident() == self.owner
        self.calls.append(threading.get_ident())

    def session_open(self, config, endpoint_only):
        self.check()
        self.open_entered.set()
        self.open_gate.wait(3)
        self.next_id += 1
        self.sessions.add(self.next_id)
        return self.next_id

    def session_close(self, session):
        self.check()
        self.sessions.discard(session)

    def submit(self, audio, rate, **kw):
        self.check()
        if rate <= 0:
            raise NativeError(1, 'bad rate')
        self.next_id += 1
        self.requests[self.next_id] = (kw, audio.copy())
        self.results.append(self.next_id)
        return self.next_id

    def poll(self, wait_ms=0):
        self.check()
        if self.fail:
            raise RuntimeError('poll failed')
        if not self.gate.is_set() or not self.results:
            time.sleep(wait_ms / 1000)
            return None
        key = self.results.popleft()
        return Completion(key, 7 if key in self.canceled else 0, self.events, 'canceled')

    def cancel(self, key):
        self.check()
        self.canceled.add(key)

    def metrics(self):
        self.check()
        return {'sessions': len(self.sessions), 'submitted': len(self.requests)}

    def close(self):
        self.check()
        self.closed = True


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 3)


@pytest.fixture
def factory(monkeypatch):
    instances = []
    def create(*a, **kw):
        engine = FakeNative(*a, **kw)
        instances.append(engine)
        return engine
    monkeypatch.setattr(module, 'NativeEngine', create)
    return instances


def test_owner_thread_and_request_postprocessing(factory):
    async def run():
        client = EngineClient('unused', EngineConfig())
        await client.start()
        try:
            engine = factory[0]
            assert await client.detect(b'\0\0', 16000) == [(0., 1.)]
            await client.open_stream('s', False, {'processor':'firered.stream_vad','version':1,'parameters':{'speech_threshold':.7}})
            assert await client.stream_chunk('s', 1, b'', 16000, True) == [(0., 1., 1.)]
            await client.close_stream('s', 2)
            assert not engine.sessions
            assert engine.owner != threading.get_ident()
            assert not client._requests and client._bytes == 0
        finally:
            await client.close()
        assert engine.closed and not client._thread.is_alive()
    asyncio.run(run())


def test_canceled_native_work_holds_capacity_until_reaped(factory):
    async def run():
        client = EngineClient('unused', EngineConfig(ingress_capacity=1))
        await client.start()
        engine = factory[0]
        engine.gate.clear()
        try:
            task = asyncio.create_task(client.detect(b'\0\0', 16000))
            await until(lambda: engine.requests)
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            await until(lambda: engine.canceled)
            with pytest.raises(ResourceExhaustedError): await client.detect(b'\0\0', 16000)
            # Health uses a snapshot and remains responsive under full admission.
            assert (await client.health())['adapter_pending'] == 1
            engine.gate.set()
            await until(lambda: not client._requests)
            assert client._bytes == 0
            assert await client.detect(b'\0\0',16000)
        finally: await client.close()
    asyncio.run(run())


def test_cleanup_bypasses_full_data_admission(factory):
    async def run():
        client = EngineClient('unused', EngineConfig(ingress_capacity=1))
        await client.start()
        engine=factory[0]
        try:
            await client.open_stream('s',False)
            engine.gate.clear()
            task=asyncio.create_task(client.stream_chunk('s',1,b'\0\0',16000,False))
            await until(lambda: engine.requests)
            await client.close_stream('s')
            assert not engine.sessions and not client._session_names
            engine.gate.set()
            with pytest.raises(FailedPreconditionError): await task
        finally: await client.close()
    asyncio.run(run())


def test_cancel_during_open_reclaims_late_session(factory):
    async def run():
        client=EngineClient('unused',EngineConfig())
        await client.start()
        engine=factory[0];engine.open_gate.clear()
        try:
            task=asyncio.create_task(client.open_stream('s',False))
            await until(engine.open_entered.is_set)
            # A blocked native call cannot block the asyncio loop.
            assert (await asyncio.wait_for(client.health(),.1))['pid']
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            engine.open_gate.set()
            await until(lambda: not client._session_names and not client._requests)
            assert not engine.sessions
            await client.open_stream('s',False)
            assert len(engine.sessions)==1
        finally:
            engine.open_gate.set();await client.close()
    asyncio.run(run())


def test_input_and_result_bytes_are_bounded(factory):
    async def run():
        client=EngineClient('unused',EngineConfig(max_inflight_mb=1))
        await client.start();engine=factory[0];engine.gate.clear()
        try:
            task=asyncio.create_task(client.detect(b'\0'*600000,16000))
            await until(lambda: engine.requests)
            with pytest.raises(ResourceExhaustedError): await client.detect(b'\0'*600000,16000)
            engine.gate.set();await task
            engine.events=[(0.,1.,1.)]*8000
            with pytest.raises(ResourceExhaustedError,match='result capacity'): await client.detect(b'\0\0',16000)
            assert client._bytes==0 and not client._requests
        finally: await client.close()
    asyncio.run(run())


def test_timeout_cancels_native_request_and_invalidates_session(factory):
    async def run():
        client=EngineClient('unused',EngineConfig(queue_timeout_ms=20))
        await client.start();engine=factory[0]
        try:
            await client.open_stream('s',False)
            engine.gate.clear()
            with pytest.raises(asyncio.TimeoutError): await client.stream_chunk('s',1,b'\0\0',16000,False)
            await until(lambda: not engine.sessions)
            with pytest.raises(FailedPreconditionError): await client.stream_chunk('s',2,b'',16000,True)
            engine.gate.set()
            await until(lambda: not client._requests)
        finally: await client.close()
    asyncio.run(run())


def test_failure_fails_pending_and_closes_engine(factory):
    async def run():
        client=EngineClient('unused',EngineConfig())
        await client.start();engine=factory[0];engine.gate.clear()
        task=asyncio.create_task(client.detect(b'\0\0',16000))
        await until(lambda: engine.requests)
        engine.fail=True
        with pytest.raises(UnavailableError,match='poll failed'): await task
        await client.close()
        assert client.failure and engine.closed and not client._requests
        with pytest.raises(UnavailableError): await client.health()
    asyncio.run(run())


def test_close_cancels_pending_and_is_idempotent(factory):
    async def run():
        client=EngineClient('unused',EngineConfig())
        await client.start();engine=factory[0];engine.gate.clear()
        task=asyncio.create_task(client.detect(b'\0\0',16000))
        await until(lambda: engine.requests)
        await client.close()
        with pytest.raises(UnavailableError): await task
        await client.close()
        assert not client._requests and client._bytes==0
    asyncio.run(run())


def test_startup_failure_and_timeout_are_cleaned(monkeypatch):
    async def run():
        def fail(*a,**kw): raise ValueError('bad model')
        monkeypatch.setattr(module,'NativeEngine',fail)
        client=EngineClient('bad',EngineConfig())
        with pytest.raises(InvalidArgumentError,match='bad model'): await client.start()
        await client.close()
        gate=threading.Event();engines=[]
        def slow(*a,**kw):
            gate.wait(3);engine=FakeNative();engines.append(engine);return engine
        monkeypatch.setattr(module,'NativeEngine',slow)
        client=EngineClient('slow',EngineConfig(startup_timeout_s=.02))
        try:
            with pytest.raises(asyncio.TimeoutError): await client.start()
        finally:
            gate.set();await client.close()
        assert engines[0].closed
    asyncio.run(run())


def test_old_completion_cannot_close_reopened_session(factory):
    async def run():
        client=EngineClient('unused',EngineConfig())
        await client.start();engine=factory[0]
        try:
            await client.open_stream('same',False)
            engine.gate.clear()
            task=asyncio.create_task(client.stream_chunk('same',1,b'\0\0',16000,False))
            await until(lambda: engine.requests)
            await client.close_stream('same')
            await client.open_stream('same',False)
            engine.gate.set()
            with pytest.raises(FailedPreconditionError): await task
            assert len(engine.sessions)==1
            assert await client.stream_chunk('same',1,b'',16000,True)
        finally: await client.close()
    asyncio.run(run())


def test_shutdown_timeout_does_not_destroy_in_use_handle(factory):
    async def run():
        client=EngineClient('unused',EngineConfig(shutdown_grace_s=.02))
        await client.start();engine=factory[0];engine.open_gate.clear()
        task=asyncio.create_task(client.open_stream('blocked',False))
        await until(engine.open_entered.is_set)
        try:
            with pytest.raises(TimeoutError,match='shutdown grace'): await client.close()
            assert not engine.closed
        finally:
            engine.open_gate.set();await client.close()
        await task
        assert engine.closed
    asyncio.run(run())
