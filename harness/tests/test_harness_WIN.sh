#!/usr/bin/env bash
# Ubuntu WSL2 tests. Temporary data/builds stay outside the repo and are cleaned.
set -euo pipefail
export WIN_TEST_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
python3 - <<'PY'
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest

repo = Path(os.environ['WIN_TEST_REPO'])

def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='dsc180-win-tests-')
        cls.root = Path(cls.temp.name)
        os.environ['DSC180_PRIVATE_WIN'] = str(cls.root / 'private')
        os.environ['DSC180_WORK_WIN'] = str(cls.root / 'work')
        cls.stage = load('stage_win_tests', repo / 'pipeline/stage_WIN.py')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.stage.prepare_data(42)
        cls.h = load('harness_win_tests', repo / 'harness/run_validation_WIN.py')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def baseline(self, name, **options):
        workspace = self.root / name
        shutil.copytree(repo / 'harness/tests/baseline_learner', workspace / 'learner')
        return self.h.Validator(workspace, name, **options)

    def shell_learner(self, name, describe='stage=1 weak=1\nstage=2 weak=1'):
        workspace = self.root / name
        learner = workspace / 'learner'
        learner.mkdir(parents=True)
        (learner / 'Makefile').write_text('all:\n\tsh probe.sh make\n\tcp train.sh train\n\tcp predict.sh predict\n\tchmod +x train predict\n')
        key = self.root / 'private/staged/validate/labels.txt'
        # These checks run from make, train, and predict, with private data under /tmp.
        probe = ('#!/bin/sh\nset -eu\n'
                 f'for p in "{key}" "{repo}/.git/HEAD" "{Path.home()}" /mnt/c/Users; do\n'
                 '  if test -r "$p"; then echo LEAKED; exit 91; fi\ndone\n'
                 'python3 -c "import socket; s=socket.socket(); s.settimeout(1); '
                 'assert s.connect_ex((\'1.1.1.1\',80)) != 0"\n'
                 'if test -d /input; then if touch /input/SHOULD_NOT_WRITE 2>/dev/null; then exit 92; fi; fi\n'
                 'echo BLOCKED > "probe_$1.txt"\n')
        (learner / 'probe.sh').write_text(probe)
        (learner / 'train.sh').write_text('#!/bin/sh\nset -eu\nsh probe.sh train\nprintf model > "$3"\n')
        (learner / 'predict.sh').write_text(
            '#!/bin/sh\nset -eu\nif test "$1" = --describe; then\n'
            + "printf '%s\\n' '" + describe + "'\nexit 0\nfi\n"
            + 'sh probe.sh predict\nfor p in "$2"/*.pgm; do id=${p##*/}; printf "%s 1.25\\n" "${id%.pgm}"; done\n')
        return self.h.Validator(workspace, name, require_cascade=True)

    def test_staging_hashes(self):
        expected = {'train': ('ada12695704433989298955081028bbde0aaeb457ea820a27449a00e44d37001',
                              '3d14b1331f3fdd397cd70fbe820b0763544da90a71cd2fd6cbe99c07b23bf665'),
                    'validate': ('74d414880a2db3b13f66aead27ee2584e8dce4b829b12e6012d0168e7c8be40a',
                                 '152233842af3700f8ebb2270d73d3bd936319d53ab11ac984e579fd03c530243')}
        for split, hashes in expected.items():
            directory = self.root / 'private/staged' / split
            self.assertEqual(hashlib.sha256((directory / 'labels.txt').read_bytes()).hexdigest(), hashes[0])
            pixels = hashlib.sha256()
            for image in sorted((directory / 'images').glob('*.pgm')):
                pixels.update(image.read_bytes())
            self.assertEqual(pixels.hexdigest(), hashes[1])

    def test_personal_workspace_contains_train_only_and_local_mcp(self):
        import tomllib
        with contextlib.redirect_stdout(io.StringIO()):
            self.stage.build_workspace('personal', 'run-01')
        workspace = self.root / 'work/personal/run-01'
        self.assertFalse(list((workspace / 'learner').iterdir()))
        self.assertEqual(len(list((workspace / 'train/images').glob('*.pgm'))), 5931)
        self.assertFalse((workspace / 'train/ledger.tsv').exists())
        self.assertFalse((workspace / 'validate').exists())
        self.assertIn('WaldBoost', (workspace / 'AGENTS.md').read_text())
        config = tomllib.loads((workspace / '.codex/config.toml').read_text())
        server = config['mcp_servers']['validation']
        self.assertEqual(server['command'], 'wsl.exe')
        self.assertIn(str(repo / 'harness/run_validation_WIN.py'), server['args'])
        self.assertIn('--require-cascade', server['args'])
        launcher = (workspace / 'start-codex.ps1').read_text()
        self.assertIn('param([switch]$Check)', launcher)
        self.assertIn('--sandbox workspace-write', launcher)
        self.assertIn('mcp_servers.validation.args=', launcher)

    def test_baseline_and_persistent_budget(self):
        validator = self.baseline('baseline', max_calls=1)
        result = validator.run()
        self.assertIn('364 faces, 682 non-faces, AUC 0.973', result)
        self.assertIn('us CPU per image', result)
        self.assertIn('predict batch: 1046 images', result)
        self.assertIn('model size:', result)
        self.assertIn('peak resident memory:', result)
        self.assertIn('budget used up', self.h.Validator(validator.workspace, 'baseline', max_calls=1).run())

    def test_filesystem_network_cascade_and_mcp(self):
        validator = self.shell_learner('isolation')
        self.assertIn('cascade: 2 stages', validator.run())
        for step in ('make', 'train', 'predict'):
            self.assertEqual((validator.run_dir / 'build_001' / f'probe_{step}.txt').read_text().strip(), 'BLOCKED')
        requests = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05'}},
                    {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                    {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                    {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'run_validation', 'arguments': {}}}]
        result = subprocess.run(['python3', str(repo / 'harness/run_validation_WIN.py'), str(validator.workspace),
                                 '--run-name', 'isolation', '--require-cascade', '--serve'],
                                input=''.join(json.dumps(r) + '\n' for r in requests), capture_output=True, text=True, check=True)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r['id'] for r in replies], [1, 2, 3])
        self.assertEqual(replies[1]['result']['tools'][0]['name'], 'run_validation')
        self.assertIn('[validation call 2/20]', replies[2]['result']['content'][0]['text'])

    def test_symlink_rejected_before_snapshot(self):
        validator = self.baseline('symlink')
        (validator.workspace / 'learner/key').symlink_to(validator.val_key)
        self.assertIn('symlinks are not allowed', validator.run())

    def test_learner_created_output_symlink_cannot_truncate_key(self):
        validator = self.shell_learner('output_symlink')
        key_before = validator.val_key.read_bytes()
        (validator.workspace / 'learner/train.sh').write_text(
            f'#!/bin/sh\nprintf model > "$3"\nln -s "{validator.val_key}" scores.txt\n')
        self.assertIn('created a symlink', validator.run())
        self.assertEqual(validator.val_key.read_bytes(), key_before)

    def test_bad_stage_numbers_and_zero_counts(self):
        for index, layout in enumerate(('stage=0 weak=1\nstage=2 weak=1', 'stage=1 weak=0\nstage=2 weak=1')):
            self.assertIn('cascade required', self.shell_learner(f'badstage{index}', layout).run())

    def test_failed_speed_run_rejected(self):
        validator = self.baseline('speedfail')
        validator._sandboxed = lambda *a, **k: (types.SimpleNamespace(returncode=1), 0, 0)
        with self.assertRaisesRegex(self.h.LearnerError, 'failed during timing'):
            validator._predict_speed(self.root, {})

    def test_cpu_timing_includes_sandboxed_workload(self):
        validator = self.baseline('cpuaccounting')
        build = validator.run_dir / 'cpu_probe'
        build.mkdir()
        result, wall, cpu = validator._sandboxed(
            'predict', ['python3', '-c', 'sum(i*i for i in range(10000000))'], build, build)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(cpu, 0.1)
        self.assertLess(cpu, wall + 0.1)

    def test_peak_memory_measures_the_learner_workload(self):
        validator = self.baseline('peakmemory')
        build = validator.run_dir / 'memory_probe'
        build.mkdir()
        record = {}
        result, _, _ = validator._sandboxed('predict',
            ['python3', '-c', 'buffer=bytearray(50*1024*1024); print(len(buffer))'],
            build, build, record)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(record['timing']['predict']['peak_rss_kib'], 45000)

    def test_evaluator_existing_fixtures_and_crosscheck(self):
        validator = self.baseline('evalcheck')
        for case in sorted((repo / 'eval/tests/cases').iterdir()):
            result = subprocess.run([str(validator.eval_bin), str(case / 'scores.txt'), str(case / 'key.txt'),
                                     *(case / 'args').read_text().split()], capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), (case / 'expected.txt').read_text().strip())
        for case in sorted((repo / 'eval/tests/bad').iterdir()):
            result = subprocess.run([str(validator.eval_bin), str(case / 'scores.txt'), str(case / 'key.txt'),
                                    '--fp-counts', '1'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(result.stdout)
            self.assertIn((case / 'expect_err').read_text().strip(), result.stderr)
        subprocess.run(['python3', str(repo / 'eval/tests/reference.py'), 'crosscheck', str(validator.eval_bin),
                        '--trials', '500'], check=True, stdout=subprocess.DEVNULL)

unittest.main(verbosity=2)
PY
