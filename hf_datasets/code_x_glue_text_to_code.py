import datasets
import os
from .common import *
import json
from .code_x_glue_code_to_text import CodeXGlueCTCodeToTextBase

class CodeXGlueTCTextToCode(Child):
    _DESCRIPTION = """The dataset we use is crawled and filtered from Microsoft Documentation, whose document located at https://github.com/MicrosoftDocs/."""
    _CITATION = """@article{iyer2018mapping,
  title={Mapping language to code in programmatic context},
  author={Iyer, Srinivasan and Konstas, Ioannis and Cheung, Alvin and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:1808.09588},
  year={2018}
}"""

    FEATURES = {
        "id": datasets.Value("int32"), # Index of the sample
        "nl": datasets.Value("string"), # The natural language description of the task
        "code": datasets.Value("string"), # The programming source code for the task
    }

    SPLITS = {"train": datasets.Split.TRAIN, "dev": datasets.Split.VALIDATION, "test": datasets.Split.TEST}

    def generate_urls(self, split_name):
        yield "data", f"concode/{split_name}.json"

    def _generate_examples(self, split_name, file_pathes):
        with open(file_pathes["data"]) as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                entry["id"] = idx
                yield idx, entry

class CodeXGlueTCNLCodeSearchAdv(CodeXGlueCTCodeToTextBase):
    LANGUAGE="python"
    SINGLE_LANGUAGE = True

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the sample
        "repo": datasets.Value("string"), # repo: the owner/repo
        "path": datasets.Value("string"),  # path: the full path to the original file
        "func_name": datasets.Value("string"),  # func_name: the function or method name
        "original_string": datasets.Value("string"),  # original_string: the raw string before tokenization or parsing
        "language": datasets.Value("string"),  # language: the programming language
        "code": datasets.Value("string"),  # code/function: the part of the original_string that is code
        "code_tokens": datasets.features.Sequence(datasets.Value("string")), # code_tokens/function_tokens: tokenized version of code
        "docstring": datasets.Value("string"),  # docstring: the top-level comment or docstring, if it exists in the original string
        "docstring_tokens": datasets.features.Sequence(datasets.Value("string")), # docstring_tokens: tokenized version of docstring
        "sha": datasets.Value("string"),  # sha of the file
        "url": datasets.Value("string"),  # url of the file
        'docstring_summary': datasets.Value("string"),  # Summary of the docstring
        'parameters' : datasets.Value("string"),  # parameters of the function
        'return_statement' : datasets.Value("string"),  # return statement
        'argument_list' : datasets.Value("string"),  # list of arguments of the function
        'identifier' : datasets.Value("string"),  # identifier
        'nwo': datasets.Value("string"),  # nwo
        'score':datasets.Value("float"),  # score for this search
    }

    def post_process(self, split_name, language, js):
        for suffix in "_tokens", "":
            key = "function" + suffix
            if key in js:
                js["code" + suffix] = js[key]
                del js[key]

        for key in self.FEATURES:
            if key not in js:
                if key =="score":
                    js[key] = -1
                else:
                    js[key]= ""

        return js

    def generate_urls(self, split_name):
        for e in super().generate_urls(split_name, self.LANGUAGE):
            yield e

    def get_data_files(self, split_name, file_pathes, language):
        if split_name == "train":
            return super().get_data_files(split_name, file_pathes, language)
        else:
            data_set_path = file_pathes["dataset"]
            data_file = os.path.join(data_set_path, "dataset", 'test_code.jsonl')
            return [data_file]

    def _generate_examples(self, split_name, file_pathes):
        for e in super()._generate_examples(split_name, file_pathes, self.LANGUAGE):
            yield e


class CodeXGlueTCNLCodeSearchWebQuery(CodeXGlueCTCodeToTextBase):
    _DESCRIPTION = """Here we present NL-code-search-WebQuery dataset, a testing set of Python code search of 1,046 query-code pairs with code search intent and their human annotations. The realworld user queries are collected from Bing query logs and the code for queries are from CodeSearchNet. You can find our testing set in ./data/test_webquery.json .
Since there's no direct training set for our WebQueryTest set, we finetune the models on an external training set by using the documentation-function pairs in the training set o fCodeSearchNet AdvTest as positive instances. For each documentation, we also randomly sample 31 more functions to form negative instances. You can run the following command to download and preprocess the data:"""

    # For each file, each line in the uncompressed file represents one function.
    # repo: the owner/repo
    # path: the full path to the original file
    # func_name: the function or method name
    # original_string: the raw string before tokenization or parsing
    # language: the programming language
    # code/function: the part of the original_string that is code
    # code_tokens/function_tokens: tokenized version of code
    # docstring: the top-level comment or docstring, if it exists in the original string
    # docstring_tokens: tokenized version of docstring
    FEATURES = {
        "id": datasets.Value("int32"),
        "repo": datasets.Value("string"),  #
        "path": datasets.Value("string"),  #
        "func_name": datasets.Value("string"),  #
        "original_string": datasets.Value("string"),  #
        "language": datasets.Value("string"),  #
        "code": datasets.Value("string"),  #
        "code_tokens": datasets.features.Sequence(datasets.Value("string")),
        "docstring": datasets.Value("string"),  #
        "docstring_tokens": datasets.features.Sequence(datasets.Value("string")),
        "sha": datasets.Value("string"),  #
        "url": datasets.Value("string"),  #
    }
    SPLITS = {"train": datasets.Split.TRAIN, "valid": datasets.Split.VALIDATION}

    LANGUAGE = "python"
    SINGLE_LANGUAGE=True

    def post_process(self, split_name, language, js):
        for suffix in "_tokens", "":
            key = "function" + suffix
            if key in js:
                js["code" + suffix] = js[key]
                del js[key]

        for key in self.FEATURES:
            if key not in js:
                if key == "score":
                    js[key] = -1
                else:
                    js[key] = ""

        return js

    def generate_urls(self, split_name):
        for key, e in super().generate_urls(split_name, self.LANGUAGE):
            if key == "language":
                yield key, e

        yield "data", f"{split_name}.txt"

    def get_data_files(self, split_name, file_pathes, language):
        if split_name == "train":
            return super().get_data_files(split_name, file_pathes, language)
        else:
            data_set_path = file_pathes["dataset"]
            data_file = os.path.join(data_set_path, "dataset", 'test_code.jsonl')
            return [data_file]

    def _generate_examples(self, split_name, file_pathes):
        import json
        import random

        random.seed(1)

        def format_str(string):
            for char in ['\r\n', '\r', '\n']:
                string = string.replace(char, ' ')
            return string

        language = self.LANGUAGE

        data_files = []
        for root, dirs, files in os.walk(language + '/final'):
            for file in files:
                temp = os.path.join(root, file)
                if '.jsonl' in temp:
                    if split_name in temp:
                        data_files.append(temp)

        data = {}
        for file in train:
            if '.gz' in file:
                os.system("gzip -d {}".format(file))
                file = file.replace('.gz', '')
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data[js['url']] = js
        print(len(data))

        urls = []
        with open("train.txt", 'r') as f1:
            for line in f1:
                line = line.strip()
                urls.append(line)
        raw_data_train = {url: data[url] for url in urls}
        print(len(raw_data_train))

        train_data = []
        idx = 1
        for url, js in raw_data_train.items():
            code = " ".join(js['code_tokens'])
            train_data.append({'idx': idx,
                               'doc': " ".join(js['docstring_tokens']),
                               'code': format_str(code),
                               'label': 1})
            idx += 1
        length = len(train_data)
        print(len(train_data))

        # num_negative = int(sys.argv[1])
        num_negative = 31
        print(num_negative)
        train_data_withneg = copy.deepcopy(train_data)
        print(len(train_data_withneg))
        for idx_x in tqdm.tqdm(range(length)):
            random_selected = random.sample(train_data[:idx_x] + train_data[idx_x + 1:length], num_negative)
            for i in range(num_negative):
                train_data_withneg.append({'idx': idx_x + length + 1,
                                           'doc': train_data[idx_x]['doc'],
                                           'code': random_selected[i]['code'],
                                           'label': 0})
        print(len(train_data_withneg))

        to_train_file = './train_codesearchnet_{}.json'.format(num_negative)
        with open(to_train_file, 'w', encoding='utf-8') as fp:
            json.dump(train_data_withneg, fp)

        data = {}
        for file in valid:
            if '.gz' in file:
                os.system("gzip -d {}".format(file))
                file = file.replace('.gz', '')
            with open(file) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data[js['url']] = js
        print(len(data))

        urls = []
        with open("valid.txt", 'r') as f1:
            for line in f1:
                line = line.strip()
                urls.append(line)
        raw_data_valid = {url: data[url] for url in urls if url in data}
        print(len(raw_data_valid))

        valid_data = []
        idx = 1
        for url, js in raw_data_valid.items():
            code = " ".join(js['code_tokens'])
            valid_data.append({'idx': idx,
                               'doc': " ".join(js['docstring_tokens']),
                               'code': format_str(code),
                               'label': 1})
            idx += 1
        print(len(valid_data))

        to_valid_file = './dev_codesearchnet.json'
        with open(to_valid_file, 'w', encoding='utf-8') as fp:
            json.dump(valid_data, fp, indent=1)