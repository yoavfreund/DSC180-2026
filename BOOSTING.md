# Image boosting prototype

## Docker

### macOS (Terminal)

Install and start Docker Desktop for the Mac's processor. From the repository
root, the following commands work in macOS Terminal (zsh or bash):

```sh
docker build -t face-boosting .
docker volume create face-boosting-auth
docker run --rm -it -p 127.0.0.1:1455:1455 \
  --mount type=volume,source=face-boosting-auth,target=/auth \
  --env BOOSTING_AUTH_DIR=/auth --env BOOSTING_AUTH_BIND=0.0.0.0 \
  --entrypoint python face-boosting -m coordinator.chatgpt_auth
```

Open the printed link and complete authorization. Then run a two-round test:

```sh
mkdir -p runs
docker run --rm \
  --mount "type=bind,source=$PWD/train,target=/data/train,readonly" \
  --mount "type=bind,source=$PWD/validate,target=/data/validate,readonly" \
  --mount "type=bind,source=$PWD/runs,target=/runs" \
  --mount type=volume,source=face-boosting-auth,target=/auth \
  --env BOOSTING_AUTH_DIR=/auth \
  face-boosting --llm-adapter /app/coordinator/llm_chatgpt.py \
  --run-dir /runs/mac-test-01 --rounds 2
```

Use a new run directory each time. Without the plan adapter and volume, the
default adapter uses an API key if provided, or falls back to fixed specs.
On Apple Silicon Docker normally builds a Linux ARM64 image; on Intel Macs it
builds Linux AMD64. The following section shows Windows PowerShell commands.

### ChatGPT Plus sign-in

Build the image and create a private Docker volume once:

```powershell
docker build -t face-boosting .
docker volume create face-boosting-auth
docker run --rm -it -p 127.0.0.1:1455:1455 --mount type=volume,source=face-boosting-auth,target=/auth --env BOOSTING_AUTH_DIR=/auth --env BOOSTING_AUTH_BIND=0.0.0.0 --entrypoint python face-boosting -m coordinator.chatgpt_auth
```

Open the printed URL in your browser, sign in, and grant plan access. Leave the
terminal running until it confirms success. Credentials stay in the Docker
volume; they are not included in the image or run logs. Protect the volume and
do not share it. This integration uses a single account registration per volume.

Then create `runs` if necessary and run the coordinator:

```powershell
New-Item -ItemType Directory -Force runs
docker run --rm --mount "type=bind,source=$($PWD.Path)/train,target=/data/train,readonly" --mount "type=bind,source=$($PWD.Path)/validate,target=/data/validate,readonly" --mount "type=bind,source=$($PWD.Path)/runs,target=/runs" --mount type=volume,source=face-boosting-auth,target=/auth --env BOOSTING_AUTH_DIR=/auth face-boosting --llm-adapter /app/coordinator/llm_chatgpt.py --run-dir /runs/plus-01 --rounds 10
```

Use a new run directory for each run. The adapter requests `gpt-6-luna` with
low reasoning effort. Account eligibility and access to that model are checked
by OpenAI when used. The local volume limits inference to 50 attempted calls in
any rolling five-hour window, including failures and timeouts; this is separate
from OpenAI's shared Plus limits and does not account for other apps. The
coordinator falls back to fixed specs if the limit is reached or requests fail.
The limit is a request cap, not a dollar spending cap. Check ChatGPT Settings →
Usage for the plan's shared allowance and app controls.

### API key or fallback

Build the container from the repository root with Docker Desktop running:

```powershell
docker build -t face-boosting .
New-Item -ItemType Directory -Force runs
docker run --rm --mount "type=bind,source=$($PWD.Path)/train,target=/data/train,readonly" --mount "type=bind,source=$($PWD.Path)/validate,target=/data/validate,readonly" --mount "type=bind,source=$($PWD.Path)/runs,target=/runs" face-boosting --run-dir /runs/experiment-01 --rounds 10
```

Choose a new run directory each time. The dataset is mounted read-only and the
models, weights and logs persist in `runs/experiment-01` on the host. For a
small smoke test, add `--rounds 2 --limit-per-class 10` (in place of the
rounds argument above). To use the OpenAI API, set `OPENAI_API_KEY` in your
shell and add `--env OPENAI_API_KEY` before `face-boosting`; otherwise the
coordinator uses its deterministic fallback. This is the separately billed
API, not ChatGPT Plus sign-in. Do not place the key in the Dockerfile.

The image defaults to `/data` and `/app/learner/libweak.so`; `--data` and
`--library` override these paths. The container builds and tests the C library
itself. The first image build downloads the base image and dependencies.

## WSL

Run in WSL from the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy 'PyJWT[crypto]==2.10.1'
make -C learner test
make -C learner
python -m unittest discover -s tests
export OPENAI_API_KEY=...  # optional; without it, fixed candidate specs are used
python main.py --run-dir runs/experiment-01 --rounds 10
```

The ChatGPT plan adapter and its tests require PyJWT with its cryptography
dependency; the standard API-key adapter uses the Python standard library.
Training also requires NumPy and GCC or another C11 compiler. Each run directory
must be new. The loader combines `train` and `validate` in memory; it does not
move files. A seeded split assigns 33% to training and 67% to validation in
each round. All rows retain weights across rounds, which are saved in the run
directory along with the models, input file order, and JSONL audit records.
The independent CBCL test set is not included or scored by this entry point.

Candidates choose pixels, LBP codes, or rectangle means derived from the C
integral-image API, and select a stump, tree, or logistic regression. The
default adapter requests specs from `gpt-6-luna` with low reasoning effort using
the separately billed API. Set `OPENAI_API_KEY` in the environment for that
adapter; otherwise it uses deterministic defaults. The ChatGPT Plus command
above explicitly selects the separate plan adapter. Credentials are not written
to the audit log. This is a bounded model-specification refinement loop, not
arbitrary code generation.
Use `--llm-adapter` to supply another provider implementation; see
[LLM_ADAPTER.md](LLM_ADAPTER.md) for the interface and Docker instructions.

Training defaults to original images plus two fixed variants: +5° rotation and
−5° rotation with a one-pixel horizontal shift, using nearest-neighbor sampling
and edge padding. The split is by original image; each training original's weight
is divided equally among its variants. `--no-augmentation` provides a baseline.
The original validation images determine selection and alpha. The audit also
records the mean weighted validation error on the two transformed validation
views as a robustness check; it does not determine selection or weight updates.

The validation error is normalized by the validation subset's weight mass.
Candidates with error below 0.5 are eligible; among candidates within 0.005
of the lowest validation error, median per-image timing breaks the tie. Timing
uses up to 32 evenly spaced original rows, a warm-up and three repetitions, measuring
all feature extraction and prediction together; if a complete timing is not
available for every contender, the lowest error wins. Model timing includes
model loading and may vary with machine load. Candidate selection uses the
same feature extraction path for all feature sets, so shared preprocessing
is charged once per candidate in this benchmark.
The selected validation error determines alpha; all image weights update and
are normalized. The configured 55-second round deadline bounds proposal and
native training workers; image loading and coordinator-side prediction and
scoring are outside that worker deadline. The deadline implementation uses
`fork` and targets Linux/WSL.

Default settings are 10 rounds, three candidates per attempt, up to three
attempts per round, seed 42, and a 67% validation share. The coordinator stops
refining when its best candidate has validation error at most 0.49; if no
candidate achieves that threshold it may still accept the best result below
0.5 after the final attempt. If none is below 0.5, the round is skipped and
weights remain unchanged. Within an attempt, speed breaks near-ties; across
attempts the best selected candidate by error is retained.

After a fallback attempt, a second fallback in the same round stops refinement
instead of retraining the same specifications. The LLM receives a bounded
summary of aggregate audit records from prior attempts, without image weights
or labels. Each accepted learner's model path, hash and alpha appear in
`manifest.jsonl`. To load and apply an ensemble in WSL:

```sh
python3 predict_ensemble.py --run-dir runs/experiment-01 --data . --output predictions.tsv
```

To run the same prediction from PowerShell using Docker:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)/train,target=/data/train,readonly" --mount "type=bind,source=$($PWD.Path)/validate,target=/data/validate,readonly" --mount "type=bind,source=$($PWD.Path)/runs,target=/runs" --entrypoint python face-boosting /app/predict_ensemble.py --run-dir /runs/experiment-01 --data /data --library /app/learner/libweak.so --output /runs/predictions.tsv
```

The output file must not already exist. Predictions are hard labels (`1` or
`-1`) alongside paths relative to the input data root.

This applies the ensemble to the supplied image folders; it does not provide
an independent test estimate. Runs made before model references were added to
the manifest cannot be loaded with this command.

As model selection repeatedly uses this combined development pool, its
validation error is not an independent final performance estimate. A separate
test set is needed to assess generalization.
