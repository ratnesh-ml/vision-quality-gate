import json
from vision_quality import evaluate

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
