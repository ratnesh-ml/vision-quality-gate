# Vision Quality Gate

> **Portfolio demo:** [Open the Ratnesh ML Lab showcase](https://ratnesh-ml-brwn1i9o5-ratnezhsingh-6317.vercel.app)

This is my computer-vision project for learning the part that is easy to skip: a model should be allowed to say **“I am not confident enough.”**


The repository generates small grayscale inspection images containing clean surfaces, scratches, or spots. It extracts image statistics and block-level features, trains a classifier, and measures both accuracy and coverage. The coverage number tells us how often the system makes a decision instead of forcing a guess.


## Pipeline


```text
synthetic image -> feature extraction -> calibrated confidence check -> class or abstain
```


| Decision | Meaning |
| --- | --- |
| Clean | No synthetic defect pattern detected |
| Scratch | Horizontal bright defect pattern |
| Spot | Local bright circular defect pattern |
| Abstain | Confidence below the review threshold |


## Run it


```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
python -m vision_quality_gate
pytest -q
```


## Why I built it this way


I wanted a CV repo that stays runnable on a normal laptop and still talks about a real deployment concern. A future version could replace the hand-built features with a small CNN, add augmentation, compare calibration methods, and include real images only when their licence and provenance are documented.


## Limitations


The images are synthetic and the features are intentionally simple. This is an educational baseline, not a factory inspection system. The confidence threshold is a design choice and needs validation against the cost of missed defects and unnecessary manual review.


## License


MIT. See [LICENSE](LICENSE).

