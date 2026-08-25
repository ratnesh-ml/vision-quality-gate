# Vision Quality Gate

[![CI](https://github.com/ratnesh-ml/vision-quality-gate/actions/workflows/test.yml/badge.svg)](https://github.com/ratnesh-ml/vision-quality-gate/actions/workflows/test.yml)

I built this computer-vision baseline to practise a behaviour that is easy to skip in student demos: a system should be able to say **“I am not confident enough to decide.”** Instead of forcing every image into a class, this project measures both the quality of its classifications and the coverage it achieves after an abstention threshold is applied.

The images are synthetic grayscale inspection examples with clean surfaces, scratches, and spots. That keeps the repository runnable on a normal laptop and lets me focus on the decision policy rather than imply that I have built a factory inspection product.

## At a glance

| Question I explored | Implementation |
| --- | --- |
| What should happen below a confidence threshold? | Route the image to **Abstain** rather than force a class. |
| How can a small CV baseline stay inspectable? | Use image statistics and block-level features instead of a large opaque model. |
| How should I judge the trade-off? | Report accuracy alongside coverage and evaluate multiple thresholds. |
| How do I keep it reproducible? | Generate deterministic synthetic images, package the code, and test it in CI. |

## Pipeline

```text
synthetic image → feature extraction → calibrated confidence check → class or abstain
```

| Decision | Meaning in this demo |
| --- | --- |
| `Clean` | No synthetic defect pattern was detected. |
| `Scratch` | A horizontal bright-defect pattern was detected. |
| `Spot` | A local bright circular pattern was detected. |
| `Abstain` | Confidence fell below the review threshold. |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m vision_quality_gate
pytest -q
```

## The engineering choice behind the project

I chose a compact feature-based baseline because I wanted the thresholding and failure behaviour to be easy to inspect. The repository exposes a configurable confidence threshold and an `evaluate_thresholds` helper so I can see the coverage-versus-abstention trade-off instead of presenting one arbitrary operating point as the answer.

## Limits I am explicit about

This is an educational baseline with synthetic images and intentionally simple features. The review threshold is a product and risk decision, not a universal constant. A real inspection workflow would need permission-cleared images, defect costs, human-review feedback, calibration analysis, and domain validation.

My next technical step would be a small CNN with documented data provenance, augmentation experiments, and a comparison of calibration methods.

## Verification and license

Run `pytest -q` for the local regression suite. GitHub Actions runs that suite on pushes and pull requests. MIT licensed; see [LICENSE](LICENSE).
