"""Remeasure a frozen validated learner and score it with the shared eval.c.

Run inside Ubuntu WSL: python3 agents/surya/benchmark.py run-01
Uses the run's latest successful private build, not another learner or run.
No agent session, tuning, or additional agent-validation call is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from harness.run_validation_WIN import Validator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('invalid run ID')
    workspace = HERE / 'runs' / args.run_id
    validator = Validator(workspace, 'sirio/' + args.run_id, require_cascade=True)
    records = [json.loads(line) for line in validator.log_path.read_text().splitlines() if line.strip()]
    successful = [record for record in records if record.get('status') == 'ok']
    if not successful:
        raise SystemExit('No successful hidden validation exists for this run')
    validated = successful[-1]
    frozen = validator.run_dir / f"build_{validated['call']:03d}"
    if not (frozen / 'model').is_file():
        raise SystemExit('No frozen validated model exists for this run')
    report = {'source': 'eval/eval.c', 'eval_source_sha256': sha(REPO / 'eval/eval.c'),
              'run': 'sirio/' + args.run_id, 'frozen_build': frozen.name,
              'validation_calls_used': len(records),
              'n_train': sum(1 for _ in (validator.train_dir / 'images').glob('*.pgm')),
              'n_predict': validator.n_val_images,
              'timing_method': 'Bubblewrap --as-pid-1; wall perf_counter; CPU RUSAGE_CHILDREN user+system',
              'predict_batch_includes': 'startup, model loading, image IO, classification, score output'}
    with tempfile.TemporaryDirectory(prefix='benchmark_', dir=validator.run_dir) as temp:
        root = Path(temp)
        build = root / 'build'
        shutil.copytree(frozen, build)
        timings = {}
        result, wall, cpu = validator._sandboxed('train', [
            './train', str(validator.train_dir / 'images'), str(validator.train_dir / 'labels.txt'),
            str(build / 'model')], build, validator.train_dir, timings)
        if result.returncode:
            raise SystemExit('Timed training failed: ' + result.stderr)
        report['train'] = {'wall_s': wall, 'cpu_s': cpu,
                           'peak_rss_kib': timings['timing']['train']['peak_rss_kib']}
        report['retrained_model_matches_validated_model'] = sha(build / 'model') == sha(frozen / 'model')
        if not report['retrained_model_matches_validated_model']:
            raise SystemExit('Retraining changed the frozen model; refusing to combine results')
        scores = root / 'scores.txt'
        with scores.open('w') as output:
            result, wall, cpu = validator._sandboxed('predict', [
                './predict', str(build / 'model'), str(validator.val_images)], build,
                validator.val_images, timings, stdout=output)
        if result.returncode:
            raise SystemExit('Timed prediction failed; validation stderr withheld')
        report['predict_batch'] = {'wall_s': wall, 'cpu_s': cpu,
                                   'peak_rss_kib': timings['timing']['predict']['peak_rss_kib']}
        report['model_bytes'] = (build / 'model').stat().st_size
        report['cascade'] = validator._describe(build, timings)
        if report['cascade']:
            report['total_classifiers'] = sum(report['cascade'])
            report['weak_evaluated'] = validator._weak_evaluated(scores, report['cascade'], {})
        evaluated = subprocess.run([str(validator.eval_bin), str(scores), str(validator.val_key),
                                    '--fp-rates', '0.01,0.05,0.1'], capture_output=True, text=True, check=True)
        report['eval_output'] = evaluated.stdout
        original = validated['eval']
        report['evaluation_matches_original'] = original == evaluated.stdout
        report['predict_timing_repeats'] = {}
        for name, images in (('full', validator.val_images), ('one', validator.one_image)):
            measurements = []
            for _ in range(3):
                result, wall, cpu = validator._sandboxed('predict', [
                    './predict', str(build / 'model'), str(images)], build, images,
                    stdout=subprocess.DEVNULL)
                if result.returncode:
                    raise SystemExit('Prediction timing repeat failed')
                measurements.append({'wall_s': wall, 'cpu_s': cpu})
            report['predict_timing_repeats'][name] = measurements
        slope = {}
        for kind in ('wall_s', 'cpu_s'):
            full = min(m[kind] for m in report['predict_timing_repeats']['full'])
            one = min(m[kind] for m in report['predict_timing_repeats']['one'])
            slope[kind.replace('_s', '_us_per_image')] = max(full - one, 0) * 1e6 / (validator.n_val_images - 1)
        report['predict_startup_subtracted'] = slope
    if not report['evaluation_matches_original']:
        raise SystemExit('Evaluation changed; refusing to report it as the original result')
    (workspace / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
