import datasets
from .common import *

class CodeXGlueTTTextToText(Child):
    _DESCRIPTION = """The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/."""

    FEATURES = {
        "id": datasets.Value("int32"), # The index of the sample
        "source": datasets.Value("string"),  # The source language version of the text
        "target": datasets.Value("string"),  # The target language version of the text
    }

    KEYS = ["source", "target"]

    SPLITS = {"train": datasets.Split.TRAIN, "dev": datasets.Split.VALIDATION, "test": datasets.Split.TEST}

    def generate_urls(self, split_name):
        lang_pair = self.info["parameters"]["natural_language_pair"]
        for i, lang in enumerate(lang_pair.split("-")):
            yield self.KEYS[i], f"{split_name}/{lang_pair}.{split_name}.{lang}"

    def _generate_examples(self, split_name, file_pathes):
        print(file_pathes)
        # Open each file (one for source language and the other for target language)
        files = {k: open(file_pathes[k]) for k in file_pathes}

        id_ = 0
        while True:
            # Read a single line from each file
            entries = {k: files[k].readline() for k in file_pathes}

            empty = self.check_empty(entries)
            if empty:
                # We are done: end of files
                return

            entries["id"] = id_
            yield id_, entries
            id_ += 1