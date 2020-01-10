import logging
from typing import Dict, Optional

from ..intent import Intent

class _IntentClassifier:
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(f"six_nlu.parser.classifier.{self.name}")
        self.logger.setLevel(logging.INFO)

    def train(self, intents: Dict[str, Intent]) -> None:
        raise NotImplementedError
    
    def reset(self):
        pass

    def parse(self, statement: str) -> Optional[str]:
        raise NotImplementedError