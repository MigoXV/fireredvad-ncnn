"""Benchmark an existing gRPC service without changing its configuration.

poetry run python scripts/benchmark_service.py --address 127.0.0.1:50022 \
  --server-pid HOST_PID --output reports/new-run --client-cpus 4,5,6,7
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from fireredvad_ncnn.benchmark import Server, measure, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--server-pid', type=int, required=True, help='Host PID for resource sampling')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wav', type=Path, default=Path('model-bin/speech.wav'))
    parser.add_argument('--client-cpus', default='4,5,6,7')
    parser.add_argument('--levels', default='1,4,8,16,32')
    parser.add_argument('--label', default='service')
    args = parser.parse_args()
    cpus = [int(c) for c in args.client_cpus.split(',')]
    levels = [int(n) for n in args.levels.split(',')]
    if not levels or any(n < 1 for n in levels):
        parser.error('Concurrency must be positive')
    source, rate = sf.read(args.wav, dtype='int16')
    if rate != 16000 or source.ndim != 1 or not len(source):
        parser.error('Use non-empty mono 16kHz WAV')
    audio = np.tile(source, int(np.ceil(160000 / len(source))))[:160000].astype('<i2').tobytes()
    class Target:
        address = args.address
        process = psutil.Process(args.server_pid)
        resources = Server.resources
    target = Target()
    server_cpus = target.process.cpu_affinity()
    if set(cpus) & set(server_cpus):
        parser.error('Bind server and client to disjoint CPU sets before benchmarking')
    os.sched_setaffinity(0, cpus)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {'label': args.label, 'address': args.address, 'server_pid': args.server_pid,
        'server_cpus': server_cpus, 'client_cpus': cpus, 'platform': platform.platform(),
        'cpu': subprocess.check_output(['lscpu'], text=True), 'duration_s': 10, 'chunk_ms': 100,
        'rounds': 3, 'input_sha256': hashlib.sha256(audio).hexdigest(),
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'worktree': subprocess.check_output(['git', 'status', '--short'], text=True),
        'server_command': target.process.cmdline(), 'resource_processes': [target.process.pid,
            *[p.pid for p in target.process.children(recursive=True)]],
        'isolation': 'disjoint CPU affinity; shared host, no exclusive CPU reservation'}
    (args.output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    summaries = []
    with (args.output/'raw.jsonl').open('w') as raw:
        def record(value):
            raw.write(json.dumps(value)+'\n'); raw.flush()
        for n in levels:
            warm = stream(target.address, audio[:64000])
            if not warm['ok']:
                record({'kind':'warmup', 'concurrency':n, **warm})
                raise RuntimeError(warm)
            short = measure(target, audio[:64000], n)
            record({'kind':'precheck', **short})
            if short['errors'] or short['min_stream_x'] < 2:
                record({'kind':'stopped', 'concurrency':n, 'reason':'precheck failed or below 2x realtime'})
                break
            rows = []
            for iteration in range(3):
                row = measure(target, audio, n)
                record({'kind':'formal', 'round':iteration+1, **row})
                rows.append(row)
                if row['errors'] or row['min_stream_x'] < 2:
                    break
            percentiles = [r['completion_p50_p95_p99_s'] for r in rows if r['completion_p50_p95_p99_s']]
            summary = {'concurrency':n, 'rounds':len(rows),
                'median_throughput':statistics.median(r['aggregate_x'] for r in rows),
                'median_completion_p50_p95_p99_s':np.median(percentiles,axis=0).tolist() if percentiles else [],
                'min_stream_x':min(r['min_stream_x'] for r in rows),
                'failures':sum(r['errors'] for r in rows),
                'max_rss_mib':max(r['server_rss_bytes'] for r in rows)/1048576,
                'median_cpu_percent':statistics.median(r['server_cpu_percent'] for r in rows)}
            summaries.append(summary)
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
            if summary['failures'] or summary['min_stream_x'] < 2:
                break
        paced = stream(target.address, audio, paced=True)
        record({'kind':'paced', **paced})
        intervals = paced.pop('response_interarrival_ms', [])
        paced.pop('events', None)
        paced['response_interarrival_p50_p95_ms'] = np.percentile(intervals,[50,95]).tolist() if intervals else []
        (args.output/'paced.json').write_text(json.dumps(paced,indent=2)+'\n')
        if not paced['ok']:
            raise RuntimeError(paced)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__ == '__main__':
    main()
