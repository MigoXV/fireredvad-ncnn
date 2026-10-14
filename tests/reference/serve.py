"""Test-only historical service entrypoint for controlled migration benchmarks."""
import fireredvad_ncnn.engine.process as process
from tests.reference.worker import run_engine_worker

process.run_engine_worker = run_engine_worker

if __name__ == "__main__":
    from fireredvad_ncnn.commands.app import app
    app()
