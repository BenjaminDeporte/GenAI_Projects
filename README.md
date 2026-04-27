# GenAI Projects

This workspace contains training scripts, datasets, and saved models for medical reasoning fine-tuning experiments.

## Trackio Dashboard

Training logs are written to the local Trackio directory used by the SFT script:

`/home/benjamin.deporte/GenAI_Projects/logs/trackio`

To view the run dashboard, launch Trackio with the same local directory in the environment:

```bash
cd /home/benjamin.deporte/GenAI_Projects
TRACKIO_DIR=/home/benjamin.deporte/GenAI_Projects/logs/trackio ./.venv/bin/trackio show --project "medical-sft-reasoning"
```

Trackio picks an available localhost port automatically. In this environment it may open on `127.0.0.1:7861`, `127.0.0.1:7862`, or another free port, so use the URL printed by the command rather than assuming `7860`.

If you want to verify that the local logs are present before opening the UI:

```bash
TRACKIO_DIR=/home/benjamin.deporte/GenAI_Projects/logs/trackio ./.venv/bin/trackio status
```
