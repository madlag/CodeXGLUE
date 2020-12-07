from pathlib import Path
import json
import shutil
import os.path
import pyclbr
import copy

class Splitter():
    UNUSED_CONFIG_KEYS = "data_dir_name", "files"

    def __init__(self, src_path, dest_path):
        self.src_path = Path(src_path)
        self.dest_path = Path(dest_path)
        self.datasets = {}
        self.class_code = {}

    def gather_definition_info(self):
        from hf_datasets.generated_definitions import DEFINITIONS

        for name, definition in DEFINITIONS.items():
            print(name)
            key = definition["dir_name"]
            definition["full_name"] = name
            if key not in self.datasets:
                self.datasets[key] = dict(configurations = [])

            self.datasets[key]["configurations"].append(definition)

        for dataset in self.datasets.values():
            names = []
            for config in dataset["configurations"]:
                names.append(config["name"])
            common_prefix = os.path.commonprefix(names)
            dataset["name"] = common_prefix

    def gather_class_info(self, module_name):
        source_code_data = pyclbr.readmodule_ex(module_name, path=["code_x_glue"])
        lines = list(open(f"hf_datasets/{module_name}.py").readlines())

        for c, d in source_code_data.items():
            code_string = "".join(lines[d.lineno - 1:d.end_lineno])
            self.class_code[c] = code_string

    def gather_classes_info(self):
        kinds = ["code", "text"]
        for src in kinds:
            for dest in kinds:
                module_name = f"code_x_glue_{src}_to_{dest}"
                self.gather_class_info(module_name)

    def generate_dataset(self, dataset_name, dataset_info, dataset_path):
        #shutil.rmtree(dataset_path, ignore_errors=True)
        dataset_path.mkdir(exist_ok=True)

        for filename in ["common.py"]:
            shutil.copy(self.src_path / filename, dataset_path / filename)

        definitions = copy.deepcopy(dataset_info["configurations"])

        for d in definitions:
            for k in self.UNUSED_CONFIG_KEYS:
                del d[k]
            config_name = d["full_name"][len(dataset_info["name"]):] or "default"
            if config_name.startswith("_"):
                config_name = config_name[1:]
            d["name"] = config_name
            del d["full_name"]
            d["sizes"] = self.sizes[dataset_name[len("code_x_glue_"):]][config_name]

        definitions = {definition["name"]:definition for definition in definitions}

        with open(dataset_path / "generated_definitions.py", "w") as f:
            f.write("DEFINITIONS=" + json.dumps(definitions, indent=4, sort_keys=True))

        BASE_CLASSES = {"CodeXGlueCCCodeCompletionTokenPython":"CodeXGlueCCCodeCompletionToken",
                        "CodeXGlueCCCodeCompletionTokenJava":"CodeXGlueCCCodeCompletionToken",
                        "CodeXGlueCCClozeTestingAll":"CodeXGlueCCClozeTesting",
                        "CodeXGlueCCClozeTestingMaxmin":"CodeXGlueCCClozeTesting",
                        "CodeXGlueTCNLCodeSearchAdv" : "CodeXGlueCTCodeToTextBase",
                        "CodeXGlueTCNLCodeSearchWebQuery": "CodeXGlueCTCodeToTextBase",
                        "CodeXGlueCTCodeToText": "CodeXGlueCTCodeToTextBase",
                        }

        child_class_names = []
        class_names = []
        for d in definitions.values():
            class_name = d["class_name"]
            if class_name not in class_names:
                if class_name in BASE_CLASSES:
                    base_class = BASE_CLASSES[class_name]
                    if base_class not in class_names:
                        class_names = [base_class] + class_names
                class_names.append(class_name)
                child_class_names.append(class_name)

        IMPORTS = ["datasets", "json", "os", "os.path"]
        configs_source_code = "".join(f"import {imp}\n" for imp in IMPORTS)
        for base_class in ["Child", "TrainValidTestChild"]:
            configs_source_code += f"from .common import {base_class}\n"

        for class_name in class_names:
            configs_source_code += self.class_code[class_name]

        with open(dataset_path / "configs.py", "w") as f:
            f.write(configs_source_code)

        with open(self.src_path / "code_x_glue_template.py") as f_in:
            s = f_in.read()
            main_class_name = dataset_info["name"].split("_")
            main_class_name = "".join([main_class_name[0].upper()] + [a.capitalize() for a in main_class_name[1:]])
            main_class_name = "CodeXGlue" + main_class_name + "Main"
            s = s.replace("class CodeXGlue(", f"class {main_class_name}(")

            class_import_string = f"from .configs import {','.join(child_class_names)}\n"
            class_import_string += "CLASS_MAPPING={"

            for child_class_name in child_class_names:
                class_import_string += f"'{child_class_name}':{child_class_name},\n"

            class_import_string += "}\n"

            s = s.replace("from .configs import *", class_import_string)



            with open(dataset_path / f"{dataset_name}.py", "w") as f_out:
                f_out.write(s)

    def generate_datasets(self):
        with(open(self.src_path /  "sizes.json")) as f:
            self.sizes = json.loads(f.read())

        for dataset_info in self.datasets.values():
            dataset_name = "code_x_glue_" + dataset_info["name"]
            dataset_path = self.dest_path / dataset_name
            self.generate_dataset(dataset_name, dataset_info, dataset_path)

    def run(self):
        self.gather_definition_info()
        self.gather_classes_info()
        self.generate_datasets()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise RuntimeError("you should specify a single argument: path to the local version of https://github.com/huggingface/datasets repository")

    dataset_path = Path(sys.argv[1])
    sys.path.append(".")

    s = Splitter("hf_datasets", dataset_path / "datasets")
    s.run()



