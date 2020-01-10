from typing import List, Dict, Optional
import re

from .classifier import _IntentClassifier
from ..intent import Intent


class RegexClassifier(_IntentClassifier):
    def __init__(self):
        super(RegexClassifier, self).__init__("RegexClassifer")
        self.patterns: List[List[str, str, re.Pattern]] = []

        self.compile_args = [re.I]
        
    def train(self, intents: Dict[str, Intent]) -> None:
        self.generate_patterns(intents)
        self.logger.info(f"Finished fitting {self.name}.")

    def generate_patterns(self, intents: Dict[str, Intent]):
        if len(self.patterns) > 0:
            self.logger.warning("Overriding existing re.Patterns")
            self.patterns = []
        
        for intent in intents.values():
            regex: str = intent.gen_regex()
            regex_pattern = re.compile(regex, *self.compile_args)
            self.patterns.append([intent.name, regex, regex_pattern])

    def reset(self) -> None:
        self.logger.warning("Recompiling re.Patterns")
        pattern: List[str, str, re.Pattern]
        for pattern in self.patterns:
            pattern[2] = re.compile(pattern[1], *self.compile_args)

    def parse(self, statement: str) -> Optional[str]:
        intent_name: str
        pattern: re.Pattern
        for intent_name, _,  pattern in self.patterns:
            if pattern.match(statement):
                self.logger.debug(f"{self.name} detected intent with name {intent_name}")
                return intent_name
    
        self.logger.info(f"{self.name} did not match any intent.")
        return None