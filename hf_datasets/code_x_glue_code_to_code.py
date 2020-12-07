import os
import json
from .common import *

class CodeXGlueCCCodeToCodeTrans(TrainValidTestChild):
    _DESCRIPTION = """The dataset is collected from several public repos, including Lucene(http://lucene.apache.org/), POI(http://poi.apache.org/), JGit(https://github.com/eclipse/jgit/) and Antlr(https://github.com/antlr/).
        We collect both the Java and C# versions of the codes and find the parallel functions. After removing duplicates and functions with the empty body, we split the whole dataset into training, validation and test sets."""
    FEATURES = {
        "id": datasets.Value("int32"),
        "java": datasets.Value("string"),  # The java version of the code
        "cs": datasets.Value("string"),  # The C# version of the code
    }

    def generate_urls(self, split_name):
        for key in "cs", "java":
            yield key, f"{split_name}.java-cs.txt.{key}"

    def _generate_examples(self, split_name, file_pathes):
        """This function returns the examples in the raw (text) form."""
        # Open each file (one for java, and one for c#)
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


class CodeXGlueCCDefectDetection(TrainValidTestChild):
    _DESCRIPTION = """Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack. We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code. 
    The dataset we use comes from the paper Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks. We combine all projects and split 80%/10%/10% for training/dev/test."""
    _CITATION = """@inproceedings{zhou2019devign,
title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
booktitle={Advances in Neural Information Processing Systems},
pages={10197--10207}, year={2019}"""

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "func": datasets.Value("string"),  # The source code
        "target": datasets.Value("bool"),  #  0 or 1 (vulnerability or not)
        "project": datasets.Value("string"),
        "commit_id": datasets.Value("string"),
    }

    def generate_urls(self, split_name):
        yield "index", f"{split_name}.txt"
        yield "data", "function.json"

    def _generate_examples(self, split_name, file_pathes):
        import json

        js_all = json.load(open(file_pathes["data"]))

        index = set()
        with open(file_pathes["index"]) as f:
            for line in f:
                line = line.strip()
                index.add(int(line))

        for idx, js in enumerate(js_all):
            if idx in index:
                js["id"] = idx
                js["target"] = int(js["target"]) == 1
                yield idx, js


class CodeXGlueCCCloneDetectionBigCloneBench(TrainValidTestChild):
    _DESCRIPTION = """Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.
The dataset we use is BigCloneBench and filtered following the paper Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree."""

    _CITATION = """@inproceedings{svajlenko2014towards,
  title={Towards a big data curated benchmark of inter-project code clones},
  author={Svajlenko, Jeffrey and Islam, Judith F and Keivanloo, Iman and Roy, Chanchal K and Mia, Mohammad Mamun},
  booktitle={2014 IEEE International Conference on Software Maintenance and Evolution},
  pages={476--480},
  year={2014},
  organization={IEEE}
}

@inproceedings{wang2020detecting,
  title={Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree},
  author={Wang, Wenhan and Li, Ge and Ma, Bo and Xia, Xin and Jin, Zhi},
  booktitle={2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering (SANER)},
  pages={261--271},
  year={2020},
  organization={IEEE}
}"""
    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "id1": datasets.Value("int32"),  # The first function id
        "id2": datasets.Value("int32"),  # The second function id
        "func1": datasets.Value("string"), # The full text of the first function
        "func2": datasets.Value("string"), # The full text of the second function
        "label": datasets.Value("bool"),  # 1 is the functions are not equivalent, 0 otherwise
    }

    def generate_urls(self, split_name):
        yield "index", f"{split_name}.txt"
        yield "data", "data.jsonl"

    def _generate_examples(self, split_name, file_pathes):
        import json

        js_all = {}

        with open(file_pathes["data"]) as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                js_all[int(entry["idx"])] = entry["func"]

        with open(file_pathes["index"]) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                idx1, idx2, label = [int(i) for i in line.split("\t")]
                func1 = js_all[idx1]
                func2 = js_all[idx2]

                yield idx, dict(id = idx, id1=idx1, id2=idx2, func1=func1, func2=func2, label=(label == 1))


class CodeXGlueCCCloneDetectionPOJ104(TrainValidTestChild):
    _DESCRIPTION = """Given a code and a collection of candidates as the input, the task is to return Top K codes with the same semantic. Models are evaluated by MAP score.
We use POJ-104 dataset on this task."""

    _CITATION = """@inproceedings{mou2016convolutional,
  title={Convolutional neural networks over tree structures for programming language processing},
  author={Mou, Lili and Li, Ge and Zhang, Lu and Wang, Tao and Jin, Zhi},
  booktitle={Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence},
  pages={1287--1293},
  year={2016}
}"""
    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "code": datasets.Value("string"), # The full text of the function
        "label": datasets.Value("string"),  # The id of problem that the source code solves
    }

    SPLIT_RANGES = {"train": (1, 65), "valid":(65,81), "test":(81, 195)}

    def generate_urls(self, split_name):
        yield "data", "programs.tar.gz"

    def pregenerate_all(self, root_path):
        def files(path):
            g = os.walk(path)
            file = []
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    file.append(os.path.join(path, file_name))
            return file

        cont = 0
        for split_name, range_info in self.SPLIT_RANGES.items():
            with open(os.path.join(root_path, f"{split_name}.jsonl"), 'w') as f:
                for i in range(*range_info):
                    items = files(os.path.join(root_path, "ProgramData/{}".format(i)))
                    for item in items:
                        js = {}
                        js['label'] = item.split('/')[1]
                        js['index'] = str(cont)
                        js['code'] = open(item, encoding='latin-1').read()
                        f.write(json.dumps(js) + '\n')
                        cont += 1

    def _generate_examples(self, split_name, file_pathes):
        root_path = file_pathes["data"]

        json_file = os.path.join(root_path, f"{split_name}.jsonl")
        if not os.path.exists(json_file):
            self.pregenerate_all(root_path)

        with open(json_file) as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                idx = int(entry["index"])
                label = entry["label"]
                e = dict(id=idx, label=label, code=entry["code"])
                yield idx, e

class CodeXGlueCCClozeTesting(Child):
    _DESCRIPTION = """Cloze tests are widely adopted in Natural Languages Processing to evaluate the performance of the trained language models. The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem.
Here we present the two cloze testing datasets in code domain with six different programming languages: ClozeTest-maxmin and ClozeTest-all. Each instance in the dataset contains a masked code function, its docstring and the target word.
The only difference between ClozeTest-maxmin and ClozeTest-all is their selected words sets, where ClozeTest-maxmin only contains two words while ClozeTest-all contains 930 words."""

    _CITATION = """@article{CodeXGLUE,
  title={CodeXGLUE: An Open Challenge for Code Intelligence},
  journal={arXiv},
  year={2020},
}
@article{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
}
@article{husain2019codesearchnet,
  title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}"""

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "idx": datasets.Value("string"), # Original index in the dataset
        "nl_tokens": datasets.features.Sequence(datasets.Value("string")),  # Natural language tokens
        "pl_tokens": datasets.features.Sequence(datasets.Value("string")),  # Programming language tokens
    }

    def generate_urls(self, split_name):
        yield "data", "clozeTest.json"

    def _generate_examples(self, split_name, file_pathes):
        with open(file_pathes["data"]) as f:
            j = json.load(f)
            index = 0
            for entry in j:
                yield index, dict(id=index, idx=entry["idx"], nl_tokens=entry["nl_tokens"], pl_tokens=entry["pl_tokens"])
                index += 1

class CodeXGlueCCClozeTestingAll(CodeXGlueCCClozeTesting):
    pass

class CodeXGlueCCClozeTestingMaxmin(CodeXGlueCCClozeTesting):
    pass


class CodeXGlueCCCodeCompletionLine(Child):
    _DESCRIPTION = """Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.
We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.
Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion."""

    _CITATION = """@article{raychev2016probabilistic,
  title={Probabilistic Model for Code with Decision Trees},
  author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
  journal={ACM SIGPLAN Notices},
  pages={731--747},
  year={2016},
  publisher={ACM New York, NY, USA}
}
@inproceedings{allamanis2013mining,
  title={Mining Source Code Repositories at Massive Scale using Language Modeling},
  author={Allamanis, Miltiadis and Sutton, Charles},
  booktitle={2013 10th Working Conference on Mining Software Repositories (MSR)},
  pages={207--216},
  year={2013},
  organization={IEEE}
}"""

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "input": datasets.Value("string"),  # Input code string
        "gt": datasets.Value("string"),  # Code string to be predicted
    }

    def generate_urls(self, split_name):
        yield "data", "test.json"

    def _generate_examples(self, split_name, file_pathes):
        with open(file_pathes["data"]) as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                entry["id"] = idx
                yield idx, entry

class CodeXGlueCCCodeCompletionLine(Child):
    _DESCRIPTION = """Complete the unfinished line given previous context. Models are evaluated by exact match and edit similarity.
We propose line completion task to test model's ability to autocomplete a line. Majority code completion systems behave well in token level completion, but fail in completing an unfinished line like a method call with specific parameters, a function signature, a loop condition, a variable definition and so on. When a software develop finish one or more tokens of the current line, the line level completion model is expected to generate the entire line of syntactically correct code.
Line level code completion task shares the train/dev dataset with token level completion. After training a model on CodeCompletion-token, you could directly use it to test on line-level completion."""

    _CITATION = """@article{raychev2016probabilistic,
  title={Probabilistic Model for Code with Decision Trees},
  author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
  journal={ACM SIGPLAN Notices},
  pages={731--747},
  year={2016},
  publisher={ACM New York, NY, USA}
}
@inproceedings{allamanis2013mining,
  title={Mining Source Code Repositories at Massive Scale using Language Modeling},
  author={Allamanis, Miltiadis and Sutton, Charles},
  booktitle={2013 10th Working Conference on Mining Software Repositories (MSR)},
  pages={207--216},
  year={2013},
  organization={IEEE}
}"""

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "input": datasets.Value("string"),  # Input code string
        "gt": datasets.Value("string"),  # Code string to be predicted
    }

    def generate_urls(self, split_name):
        yield "data", "test.json"

    def _generate_examples(self, split_name, file_pathes):
        with open(file_pathes["data"]) as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                entry["id"] = idx
                yield idx, entry

class CodeXGlueCCCodeCompletionToken(Child):
    _DESCRIPTION = """Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.
    Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.
    """

    _CITATION = """@article{raychev2016probabilistic,
      title={Probabilistic Model for Code with Decision Trees},
      author={Raychev, Veselin and Bielik, Pavol and Vechev, Martin},
      journal={ACM SIGPLAN Notices},
      pages={731--747},
      year={2016},
      publisher={ACM New York, NY, USA}
    }
    @inproceedings{allamanis2013mining,
      title={Mining Source Code Repositories at Massive Scale using Language Modeling},
      author={Allamanis, Miltiadis and Sutton, Charles},
      booktitle={2013 10th Working Conference on Mining Software Repositories (MSR)},
      pages={207--216},
      year={2013},
      organization={IEEE}
    }"""


class CodeXGlueCCCodeCompletionTokenJava(CodeXGlueCCCodeCompletionToken):
    SPLITS = {"training": datasets.Split.TRAIN, "validation": datasets.Split.VALIDATION, "test": datasets.Split.TEST}

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "code": datasets.features.Sequence(datasets.Value("string")),  # Code Tokens
    }

    def generate_urls(self, split_name):
        language = self.info["parameters"]["language"]
        if language != "java":
            raise RuntimeError(f"Unknown language {language}: should be java.")

        yield "data", f"https://zenodo.org/record/3628665/files/java_{split_name}_pre"

    def _generate_examples(self, split_name, file_pathes):
        with open(file_pathes["data"]) as f:
            for idx, line in enumerate(f):
                new_data = []
                for token in line.strip().split():
                    if len(token) > 100:
                        continue
                    new_data.append(token)
                entry = dict(id=idx, code=new_data)
                yield idx, entry


class CodeXGlueCCCodeCompletionTokenPython(CodeXGlueCCCodeCompletionToken):
    SPLITS = {"train": datasets.Split.TRAIN, "test": datasets.Split.TEST}

    FEATURES = {
        "id": datasets.Value("int32"),  # Index of the example
        "path": datasets.Value("string"),  # Original path in the dataset
        "code": datasets.features.Sequence(datasets.Value("string")),  # Code Tokens
    }

    PYTHON_FILE_MAPPING = dict(train="python100k_train.txt",
                               test="python50k_eval.txt")

    def generate_urls(self, split_name):
        language = self.info["parameters"]["language"]
        if language != "python":
            raise RuntimeError(f"Unknown language {language}")

        yield "data", "http://files.srl.inf.ethz.ch/data/py150_files.tar.gz"

    def process_string(self, token):
        # Copyright (c) Microsoft Corporation.
        # Licensed under the MIT License.
        import re
        str_quote_options = ["'''", '"""', "'", '"']
        start_quote = ""
        end_quote = ""
        qualifier_regex = r"^[a-z]+"
        qualifier_match = re.search(qualifier_regex, token)
        # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
        qualifier = "" if not qualifier_match else qualifier_match[0]
        # token string without qualifiers
        token_string = re.sub(qualifier_regex, "", token)
        # string literal without quotes
        str_lit = token_string
        for q in str_quote_options:
            if token_string.startswith(q):
                start_quote = q
                str_lit = str_lit[len(q):]
                if token_string.endswith(q):
                    end_quote = q
                    str_lit = str_lit[: -len(q)]
                break
        if start_quote in str_quote_options[:2]:
            return ""
        return (
            f"{qualifier}{start_quote}{str_lit}{end_quote}"
            if len(
                str_lit) < 15 and "\n" not in str_lit and "</s>" not in str_lit and "<s>" not in str_lit and "<pad>" not in str_lit and "<EOL>" not in str_lit
            else f"{qualifier}{start_quote}{end_quote}"
        )

    def py_tokenize(self, base_dir, file_name):
        # Copyright (c) Microsoft Corporation.
        # Licensed under the MIT License.
        from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
        from io import BytesIO

        file_paths = open(os.path.join(base_dir, file_name)).readlines()
        for ct, path in enumerate(file_paths):
            try:
                code = open(os.path.join(base_dir, path.strip())).read()
                token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
                out_tokens = []
                prev_eol = False
                for toknum, tokval, _, _, _ in token_gen:
                    tokval = " ".join(tokval.split())
                    if len(tokval) > 100:
                        continue
                    if toknum == STRING:
                        add_token = self.process_string(tokval)
                        if len(add_token) > 0:
                            out_tokens.append(add_token)
                            prev_eol = False
                    elif toknum == NUMBER:
                        if len(tokval) < 50:
                            out_tokens.append(tokval)
                            prev_eol = False
                    elif toknum in [NEWLINE, NL]:
                        if not prev_eol:
                            out_tokens.append("<EOL>")
                            prev_eol = True
                    elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                        continue
                    else:
                        out_tokens.append(tokval)
                        prev_eol = False
                if out_tokens[0] == "<EOL>":
                    out_tokens = out_tokens[1:]
                if out_tokens[-1] == "<EOL>":
                    out_tokens = out_tokens[:-1]
            except Exception:
                out_tokens = []
            out_tokens = ["<s>"] + out_tokens + ["</s>"]
            yield path, out_tokens

    def _generate_examples(self, split_name, file_pathes):
        base_dir = file_pathes["data"]
        filename = self.PYTHON_FILE_MAPPING[split_name]
        if not os.path.exists(os.path.join(base_dir, "data")):
            import gzip
            import tarfile
            gzip_filename = os.path.join(base_dir, "data.tar.gz")
            with gzip.open(gzip_filename, "rb") as gzip_file:
                t = tarfile.TarFile(fileobj=gzip_file)
                t.extractall(path=base_dir)

        idx = 0
        for entry in self.py_tokenize(base_dir=base_dir, file_name=filename):
            path, out_tokens = entry
            path = path[len("data/"):]
            yield idx, dict(id=idx, path=path, code=out_tokens)
            idx += 1

class CodeXGlueCCCodeRefinement(TrainValidTestChild):
    _DESCRIPTION = """We use the dataset released by this paper(https://arxiv.org/pdf/1812.08693.pdf). The source side is a Java function with bugs and the target side is the refined one. All the function and variable names are normalized. Their dataset contains two subsets ( i.e.small and medium) based on the function length."""
    FEATURES = {
        "id": datasets.Value("int32"),
        "buggy": datasets.Value("string"),  # The java version of the code
        "fixed": datasets.Value("string"),  # The C# version of the code
    }

    def generate_urls(self, split_name):
        size = self.info["parameters"]["size"]
        for key in "buggy", "fixed":
            yield key, f"{size}/{split_name}.buggy-fixed.{key}"

    def _generate_examples(self, split_name, file_pathes):
        """This function returns the examples in the raw (text) form."""
        # Open each file (one for java, and one for c#)
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
