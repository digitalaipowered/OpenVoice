# OpenVoice Forge

OpenVoice Forge is a production-oriented web wrapper around this repository's OpenVoice V2 inference path. It does not replace the OpenVoice model or route generation through another TTS provider.

## What it does

1. Accepts a reference voice clip.
2. Extracts the target speaker tone-color embedding with `openvoice.se_extractor.get_se`.
3. Generates base speech with MeloTTS in the selected language.
4. Loads the matching OpenVoice V2 base-speaker embedding.
5. Converts the base speech into the reference speaker's tone color with `ToneColorConverter.convert`.
6. Returns a downloadable WAV file.

Supported UI languages: English, Spanish, French, Chinese, Japanese, and Korean.

The UI requires the operator to confirm that they own the reference voice or have explicit permission to clone it.

## Run locally

OpenVoice V2 requires Python 3.9, the OpenVoice V2 checkpoints, MeloTTS, and FFmpeg.

```bash
conda create -n openvoice-forge python=3.9 -y
conda activate openvoice-forge
pip install -e .
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

Download and extract the official V2 checkpoint archive so this file exists:

```text
checkpoints_v2/converter/checkpoint.pth
```

Then launch:

```bash
python -m forge_app
```

The default port is `7860`.

## Optional private access

Set both environment variables before launch:

```bash
export OPENVOICE_FORGE_USERNAME='owner'
export OPENVOICE_FORGE_PASSWORD='choose-a-strong-password'
python -m forge_app
```

Do not commit real passwords to GitHub.

## Docker

Build the self-contained image. The Dockerfile installs OpenVoice, MeloTTS, FFmpeg, UniDic, and the official OpenVoice V2 checkpoints.

```bash
docker build -f Dockerfile.forge -t openvoice-forge .
docker run --rm -p 7860:7860 openvoice-forge
```

For private access:

```bash
docker run --rm -p 7860:7860 \
  -e OPENVOICE_FORGE_USERNAME='owner' \
  -e OPENVOICE_FORGE_PASSWORD='choose-a-strong-password' \
  openvoice-forge
```

## GPU deployment

The application automatically selects `cuda:0` when CUDA is visible to PyTorch and otherwise falls back to CPU. For production voice generation, a CUDA-capable host is strongly preferred because CPU inference can be slow.

If the container image is rebuilt on a CUDA-enabled PyTorch base instead of `python:3.9-slim`, no application-code changes are required. The runtime selection is automatic.

## Environment variables

- `OPENVOICE_CKPT_DIR` — path to the OpenVoice V2 checkpoint directory. Default: `checkpoints_v2`.
- `OPENVOICE_FORGE_OUTPUT_DIR` — generated WAV directory. Default: `outputs_forge`.
- `OPENVOICE_FORGE_PORT` — local port. Default: `7860`.
- `PORT` — hosting-platform port override.
- `OPENVOICE_FORGE_USERNAME` and `OPENVOICE_FORGE_PASSWORD` — optional Gradio basic authentication.
- `OPENVOICE_FORGE_MAX_TEXT_CHARS` — per-generation text limit. Default: `1800`.
- `OPENVOICE_FORGE_MIN_REFERENCE_SECONDS` — minimum reference duration. Default: `2`.
- `OPENVOICE_FORGE_MAX_REFERENCE_SECONDS` — maximum reference duration. Default: `60`.

## Safety and deployment notes

Keep public deployments authenticated unless you deliberately intend to operate a public voice-cloning service. Do not expose preset voices of people who have not authorized cloning. Reference audio is supplied directly to the running OpenVoice instance; configure host-level retention and storage policies appropriate for your deployment.
