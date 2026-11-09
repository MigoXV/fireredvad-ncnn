"""The native service CLI exposes only supported, effective options."""
import pytest
from typer.testing import CliRunner
from fireredvad_ncnn.commands.app import app


@pytest.mark.parametrize('option', ['--batch-size', '--batch-wait-ms', '--workers', '--chunk-size'])
def test_removed_service_options_are_rejected(option):
    result = CliRunner().invoke(app, ['serve', 'unused-model', option, '4'])
    assert result.exit_code == 2
    assert 'No such option' in result.output


def test_external_torch_benchmark_entrypoint_is_removed():
    result = CliRunner().invoke(app, ['benchmark'])
    assert result.exit_code == 2
    assert 'No such command' in result.output
