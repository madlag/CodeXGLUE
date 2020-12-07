from typing import List
import datasets

from .generated_definitions import DEFINITIONS
from .common import *
from .configs import *

class CodeXGlueConfig(datasets.BuilderConfig):
    pass

class {{CodeXGlue}}(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = CodeXGlueConfig
    BUILDER_CONFIGS = [CodeXGlueConfig(name=name, description=info["description"]) for name, info in DEFINITIONS.items()]

    def _info(self):
        name = self.config.name
        info = DEFINITIONS[name]
        if info["class_name"] in globals():
            self.child = globals()[info["class_name"]](info)
        else:
            raise RuntimeError(f"Unknown python class for dataset configuration {name}")
        ret = self.child._info()
        return ret

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return self.child._split_generators(dl_manager=dl_manager)

    def _generate_examples(self, split_name, file_pathes):
        return self.child._generate_examples(split_name, file_pathes)
