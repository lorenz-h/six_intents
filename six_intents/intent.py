import re
import random
import copy
import logging
import json
from typing import List, Any


class _Slot:

    slot_type = None

    def __init__(self, spec):
        self.name: str = spec["name"]
        self.required: bool = spec["required"]
        self.regex_placeholder: str = self.gen_regex_placeholder()

    def gen_regex_placeholder(self) -> str:
        raise NotImplementedError("_Slot is abstract")

    def get_random_value(self) -> Any:
        raise NotImplementedError("_Slot is abstract")
    
    def __iter__(self):
        for key, value in self.__dict__.items():
            yield key, value


class CategoricalSlot(_Slot):

    slot_type = "CategoricalSlot"

    def __init__(self, spec):
        self.legal_values: List[Any] = spec["legal_values"]
        super(CategoricalSlot, self).__init__(spec)

    def gen_regex_placeholder(self) -> str:
        return f"({'|'.join([str(val) for val in self.legal_values])})"

class NumericalSlot(_Slot):

    slot_type = "NumericalSlot"

    def __init__(self, spec):
        super(NumericalSlot, self).__init__(spec)
    
    def gen_regex_placeholder(self) -> str:
        return r"[-+]?[0-9]*\.?[0-9]+"

def create_slot(spec: dict) -> _Slot:
    if spec["slot_type"] == "categorical":
        return CategoricalSlot(spec)
    elif spec["slot_type"] == "Numerical":
        return NumericalSlot(spec)
    else:
        raise ValueError(f"Undefined Slot type {spec['slot_type']}")


class Intent:
    def __init__(self, spec: dict, sanitize_fn):
        self.logger = logging.getLogger("six_core.module.nlu.intent")
        self.test(spec)
        self.name = spec["name"]
        self.slots: List[_Slot] = [create_slot(slot_spec) for slot_spec in spec["slots"]]
        self.templates = [sanitize_fn(template) for template in spec["templates"]]
        self.samples = self.gen_samples()

    def gen_regex(self) -> str:
        regex_strs = []
        for template in self.templates:
            for slot in self.slots:
                template_regex_str = template.replace(f"[{slot.name}]", slot.regex_placeholder)
                regex_strs.append(template_regex_str)
        
        regex_str = '|'.join(regex_strs)
        return regex_str

    def __str__(self):
        return json.dumps({"name":self.name, "slots":[dict(slot) for slot in self.slots]})

    def gen_samples(self) -> list:
        samples = []
        for template in self.templates:
            sample = copy.copy(template)
            
            for slot in self.slots:
                sample = sample.replace(f"[{slot.name}]", str(slot.get_random_value()))

            self.logger.debug(f"created sample <{sample}>")
            samples.append(sample)
            
        return samples

    @staticmethod
    def test(spec):
        for key in ["name", "templates", "slots"]:
            try: 
                val = spec[key]
            except KeyError:
                raise KeyError(f"Intent Spec missing key <{key}>")
        assert isinstance(spec["templates"], list) and len(spec["templates"]) > 0, f"Intent {spec['name']} has no templates"

