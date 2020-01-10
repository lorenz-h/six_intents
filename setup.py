import subprocess
from utils import module_path

if __name__ == "__main__":
    module_root_dir = module_path("")
    subprocess.call(["git", "clone", "https://github.com/facebookresearch/InferSent.git"], cwd=module_root_dir)
    