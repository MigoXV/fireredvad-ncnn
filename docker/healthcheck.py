"""通过服务实际监听端口检查标准 gRPC Health。"""
import os

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

with grpc.insecure_channel(f"127.0.0.1:{os.environ.get('GRPC_PORT', '50051')}") as channel:
    response = health_pb2_grpc.HealthStub(channel).Check(
        health_pb2.HealthCheckRequest(), timeout=5
    )
    if response.status != health_pb2.HealthCheckResponse.SERVING:
        raise SystemExit(1)
