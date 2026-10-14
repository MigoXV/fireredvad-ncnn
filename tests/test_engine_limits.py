from collections import deque
from types import SimpleNamespace
import time

import numpy as np
import pytest

from fireredvad_ncnn.configs.engine_config import EngineConfig
from fireredvad_ncnn.engine.protocol import (
    FailedPreconditionError, ResourceExhaustedError, Operation, Status,
    pack_metadata, unpack_metadata,
)
from tests.reference.worker import (
    _Metrics, _Request, _Session, _Work, _enqueue_stream_request,
    _drop_expired_or_canceled, _receive_and_dispatch,
)
from tests.test_engine_protocol import _metadata


class Socket:
    def __init__(self, metadata=None):
        self.metadata = metadata
        self.responses = []

    def recv_multipart(self):
        return [b'client', pack_metadata(self.metadata), b'']

    def send_multipart(self, frames):
        self.responses.append(unpack_metadata(frames[1]))


def test_session_queue_limit_and_sequence():
    session = _Session(state=None, endpoint_only=False, busy=True)
    sessions = {'s': session}
    config = EngineConfig(session_queue_capacity=1)
    request = _Request(b'client', {'request_id':'one','session_id':'s','sequence':1,'sample_rate':16000}, b'\0\0')
    _enqueue_stream_request(None,None,'generation',config,request,sessions,[],_Metrics())
    assert len(session.queue)==1
    with pytest.raises(FailedPreconditionError):
        _enqueue_stream_request(None,None,'generation',config,request,sessions,[],_Metrics())
    request.metadata['sequence']=2
    with pytest.raises(ResourceExhaustedError):
        _enqueue_stream_request(None,None,'generation',config,request,sessions,[],_Metrics())
    assert session.expected_sequence==2


def test_max_sessions_and_expired_ingress():
    config=EngineConfig(max_sessions=1)
    metadata=_metadata(op=Operation.STREAM_OPEN.value,session_id='new',payload_length=0)
    socket=Socket(metadata)
    _receive_and_dispatch(socket,None,'generation',config,False,1024,{'existing':None},[],set(),_Metrics())
    assert socket.responses[-1]['status']==Status.RESOURCE_EXHAUSTED.value
    metadata['deadline_ns']=1
    _receive_and_dispatch(socket,None,'generation',config,False,1024,{},[],set(),_Metrics())
    assert socket.responses[-1]['status']==Status.DEADLINE_EXCEEDED.value


@pytest.mark.parametrize('cancel',[False,True])
def test_queued_cancel_or_timeout_releases_state(cancel):
    released=[]
    inferencer=SimpleNamespace(runtime=SimpleNamespace(release_session_cache=released.append))
    session=_Session(state=None,endpoint_only=False,model_cache='owned-state')
    request=_Request(b'client',{'request_id':'r','deadline_ns':1 if not cancel else time.monotonic_ns()+10**9},b'')
    work=_Work(request,'stream',deque([np.zeros((10,80))]),session_id='s')
    ready=[work]; sessions={'s':session}; canceled={'r'} if cancel else set()
    socket=Socket()
    _drop_expired_or_canceled(socket,inferencer,'generation',sessions,ready,canceled,_Metrics())
    assert not ready and not sessions and not canceled
    assert released==['owned-state']
    if cancel:
        assert not socket.responses
    else:
        assert socket.responses[-1]['status']==Status.DEADLINE_EXCEEDED.value
