import pathlib
from typing import Union

def module_path(fpath: Union[str, pathlib.Path]) -> pathlib.Path:
    module_path = pathlib.Path(__file__).parent.parent.resolve()
    return module_path / fpath