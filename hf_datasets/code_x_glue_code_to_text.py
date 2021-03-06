from typing import List
import datasets
import os
import json
from .common import *

class CodeXGlueCTCodeToTextBase(TrainValidTestChild):
    _DESCRIPTION = """The dataset we use comes from CodeSearchNet and we filter the dataset as the following:
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.
"""
    _CITATION = """@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}"""

    # For each file, each line in the uncompressed file represents one function.
    _FEATURES = {
        "id": datasets.Value("int32"), # Index of the sample
        "repo": datasets.Value("string"), # repo: the owner/repo
        "path": datasets.Value("string"),  # path: the full path to the original file
        "func_name": datasets.Value("string"),  # func_name: the function or method name
        "original_string": datasets.Value("string"),  # original_string: the raw string before tokenization or parsing
        "language": datasets.Value("string"),  # language: the programming language name
        "code": datasets.Value("string"),  # code/function: the part of the original_string that is code
        "code_tokens": datasets.features.Sequence(datasets.Value("string")), # code_tokens/function_tokens: tokenized version of code
        "docstring": datasets.Value("string"), # docstring: the top-level comment or docstring, if it exists in the original string
        "docstring_tokens": datasets.features.Sequence(datasets.Value("string")), # docstring_tokens: tokenized version of docstring
        "sha": datasets.Value("string"),  # sha of the file
        "url": datasets.Value("string"),  # url of the file
    }

    _SUPERVISED_KEYS = ["docstring", "docstring_tokens"]

    def generate_urls(self, split_name, language):
        yield "language", f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"
        yield "dataset", "dataset.zip"

    def get_data_files(self, split_name, file_paths, language):
        language_specific_path = file_paths["language"]
        final_path = os.path.join(language_specific_path, language, 'final')
        # Make some cleanup to save space
        for path in os.listdir(final_path):
            if path.endswith(".pkl"):
                os.unlink(path)

        data_files = []
        for root, dirs, files in os.walk(final_path):
            for file in files:
                temp = os.path.join(root, file)
                if '.jsonl' in temp:
                    if split_name in temp:
                        data_files.append(temp)
        return data_files

    def post_process(self, split_name, language, js):
        return js

    def _generate_examples(self, split_name, file_paths, language):
        import gzip

        data_set_path = file_paths["dataset"]

        data_files = self.get_data_files(split_name, file_paths, language)

        urls = {}
        f1_path_parts = [data_set_path, "dataset", language, f"{split_name}.txt"]
        if self.SINGLE_LANGUAGE:
            del f1_path_parts[2]

        f1_path = os.path.join(*f1_path_parts)
        with open(f1_path) as f1:
            for line in f1:
                line = line.strip()
                urls[line] = True

        idx = 0
        for file in data_files:
            if '.gz' in file:
                f = gzip.open(file)
            else:
                f = open(file)

            for line in f:
                line = line.strip()
                js = json.loads(line)
                if js['url'] in urls:
                    js["id"] = idx
                    js = self.post_process(split_name, language, js)
                    if "partition" in js:
                        del js["partition"]
                    yield idx, js
                    idx += 1
            f.close()


class CodeXGlueCTCodeToText(CodeXGlueCTCodeToTextBase):
    SINGLE_LANGUAGE = False

    def generate_urls(self, split_name):
        language = self.info["parameters"]["language"]
        for e in super().generate_urls(split_name, language):
            yield e

    def _generate_examples(self, split_name, file_paths):
        language = self.info["parameters"]["language"]
        for e in super()._generate_examples(split_name, file_paths, language):
            yield e