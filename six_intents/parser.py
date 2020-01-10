import logging
import pathlib
import json
import pickle

from typing import Union, Dict, Optional


from .intent import Intent
from .classifiers import RegexClassifier, ProbabilisticClassifier


class IntentParser:
    def __init__(self):
        self.name: str = "IntentParser"
        self.logger = logging.getLogger(f"six_nlu.parser.{self.name}")
        self.logger.setLevel(logging.INFO)
        self.sanitizer = str.maketrans('', '', r"""!"#$%&'()*+,-./:;<=>?@\^_`{|}~""")
        
        self.intents: Dict[str, Intent] = {}

        self.regex_classifier = RegexClassifier()
        self.probabilistic_classifer = ProbabilisticClassifier()

    def train(self, fpath: Union[str, pathlib.Path]) -> None:
        self.load_training_data(fpath)

        self.regex_classifier.train(self.intents)
        self.probabilistic_classifer.train(self.intents)
    
    def load_training_data(self, fpath: Union[str, pathlib.Path]) -> None:
        self.logger.info(f"Loading training data from disk...")
        
        with open(fpath, "r", encoding="utf-8") as fp:
            train_data = json.load(fp)

        self.logger.info(f"Generating Intents...")
        for intent_spec in train_data["intents"]:
            self.intents.update({intent_spec["name"]: Intent(intent_spec, self.sanitize)})

    def sanitize(self, statement):
        clean_statement = statement.translate(self.sanitizer)
        self.logger.debug(f"sanitized <{statement}> to <{clean_statement}>")
        return clean_statement

    def parse(self, statement: str) -> Optional[Intent]:
        clean_statement = self.sanitize(statement)
        proba_intent_name =  self.probabilistic_classifer.parse(clean_statement)
        return self.intents[proba_intent_name]
        """
        regex_intent_name: Optional[str] = self.regex_classifier.parse(clean_statement)
        if regex_intent_name is None:
            proba_intent_name: Optional[str] = self.probabilistic_classifer.parse(clean_statement)
            if proba_intent_name is None:
                return None
            else:
                return self.intents[proba_intent_name]
        else:
            return self.intents[regex_intent_name] 
        """

    def persist(self, directory: Union[str, pathlib.Path]) -> None:
        self.logger.info(f"Saving {self.name} to {directory}")
        fpath = pathlib.Path(directory) / f"{self.name}.pkl"
        with open(fpath, "wb") as fh:
            pickle.dump(self.__dict__, fh)

    def load(self, directory: Union[str, pathlib.Path]) -> None:
        fpath = pathlib.Path(directory) / f"{self.name}.pkl"
        with open(fpath, "rb") as fh:
            self.__dict__ = pickle.load(fh)
        
        self.probabilistic_classifer.reset()
        self.regex_classifier.reset()