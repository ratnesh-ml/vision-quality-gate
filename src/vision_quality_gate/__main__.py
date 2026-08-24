import json
from .vision import evaluate

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
