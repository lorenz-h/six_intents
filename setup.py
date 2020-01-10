import subprocess
from six_intents.utils import module_path

if __name__ == "__main__":
    module_root_dir = module_path("")
    subprocess.call(["git", "submodule", "init", "--recursive"], cwd=module_root_dir)
    