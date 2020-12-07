import json
import os
import re
from pathlib import Path


class DatasetCodeBuilder:
    def __init__(self, git_path):
        self.git_path = Path(git_path)

    def to_snake_case(self, s):
        return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", s).lower().replace("-", "_").replace("__", "_")

    def to_class_name(self, original_name):
        return "".join([o[0].upper() + o[1:] for o in original_name.split("-")])

    def run(self):
        datasets_info = {}
        DATASET_TYPES = {"Code-Code":"cc", "Code-Text":"ct", "Text-Text":"tt", "Text-Code":"tc"}
        URL = "https://github.com/madlag/CodeXGLUE/tree/main"
        RAW_URL = "https://raw.githubusercontent.com/madlag/CodeXGLUE/main"
        LANGUAGES=["go", "java", "javascript", "php", "python", "ruby"]
        TEXT_TO_TEXT_LANGUAGES_PAIR = [l+"-en" for l in ["da", "lv", "no", "zh"]]
        COMPLETION_LANGUAGES={"java":"javaCorpus", "python":"py150"}
        REFINEMENT_SIZES=["small", "medium"]

        for dataset_type, dataset_type_shortname in DATASET_TYPES.items():
            code_path = self.git_path / dataset_type
            for d in code_path.iterdir():
                normalized_name = self.to_snake_case(dataset_type_shortname + "-" + d.name)
                print(normalized_name)
                if "Cloze" in d.name:
                    for language in LANGUAGES:
                        datasets_info[normalized_name + "_" + language] = {"dataset_type":dataset_type, "dir_name": d.name, "name": normalized_name, "parameters":{"language":language}}
                elif "CodeCompletion" in d.name:
                    for language, original_language_name in COMPLETION_LANGUAGES.items():
                        datasets_info[normalized_name + "_" + language.lower()] = {"dataset_type":dataset_type, "dir_name": d.name, "name": normalized_name, "parameters":{"language":language, "original_language_name":original_language_name}}
                elif "code-refinement" in d.name:
                    for size in REFINEMENT_SIZES:
                        datasets_info[normalized_name + "_" + size.lower()] = {"dataset_type":dataset_type, "dir_name": d.name, "name": normalized_name, "parameters":{"size":size}}
                elif "code-to-text" in d.name:
                    for language in LANGUAGES:
                        datasets_info[normalized_name + "_" + language] = {"dataset_type": dataset_type,
                                                                           "dir_name": d.name,
                                                                           "name": normalized_name,
                                                                           "parameters": {"language": language}}
                elif "text-to-text" in d.name:
                    for language_pair in TEXT_TO_TEXT_LANGUAGES_PAIR:
                        datasets_info[normalized_name + "_" + language_pair.replace("-","_")] = {"dataset_type": dataset_type,
                                                                           "dir_name": d.name,
                                                                           "name": normalized_name,
                                                                           "parameters": {"natural_language_pair": language_pair}}
                else:
                    datasets_info[normalized_name] = {"dataset_type":dataset_type, "dir_name": d.name, "name": normalized_name}


        for k, v in datasets_info.items():
            base_data_dir = None
            for name in "data", "dataset", ".":
                datapath = self.git_path / v["dataset_type"] / v["dir_name"] / name
                if datapath.exists():
                    base_data_dir = name
                    break

            original_name = v["dir_name"]
            if "parameters" in v and "language" in v["parameters"]:
                language = v["parameters"]["language"]
                original_language_name = v["parameters"].get("original_language_name", language)
                if "cloze_testing_all_" in k :
                    middle_dir_name = "cloze-all"
                elif "cloze_testing_maxmin_" in k:
                    middle_dir_name = "cloze-maxmin"
                elif "code_completion_token" in k:
                    middle_dir_name = f"{original_language_name}"
                    original_language_name = None
                elif "code_completion_line" in k:
                      middle_dir_name = f"{original_language_name}/line_completion"
                      original_language_name = None
                elif "code_to_text" in k:
                      middle_dir_name = None
                      original_language_name = None
                else:
                    print(k)
                    raise RuntimeError("Unknown sub")
                path_parts = []
                for p in base_data_dir, middle_dir_name, original_language_name:
                    if p is not None and p != ".":
                        path_parts.append(p)
                if len(path_parts) == 0:
                    path_parts = ["."]
                v["data_dir_name"] = os.path.join(*path_parts)
            else:
                v["data_dir_name"] = base_data_dir

            v["raw_url"] = f"{RAW_URL}/{v['dataset_type']}/{original_name}"
            if v['data_dir_name'] != "." :
                v["raw_url"] += f"/{v['data_dir_name']}"
            url = f"{URL}/{v['dataset_type']}/{original_name}"
            v["project_url"] = url
            url_original = url.replace("madlag", "microsoft")
            v["description"] = f"CodeXGLUE {original_name} dataset, available at {url_original}"
            v["class_name"] = "CodeXGlue" + DATASET_TYPES[v["dataset_type"]].upper() + self.to_class_name(original_name)

            if "code_completion_token" in k:
                v["class_name"] += language.capitalize()

            v["files"] = []
            d = self.git_path / v["dataset_type"]  / v["dir_name"] / v["data_dir_name"]
            for root, dirs, files in os.walk(d):
                for f in files:
                    v["files"].append(os.path.join(root, f))

        with(open(os.path.join(os.path.dirname(__file__),  "test_generated.json"))) as f:
            sizes = json.loads(f.read())

        with open(os.path.join(os.path.dirname(__file__),  "generated_definitions.py"), "w") as f:
            f.write("DEFINITIONS=" + json.dumps(datasets_info, indent=4, sort_keys=True))


def main(git_path):
    dcb = DatasetCodeBuilder(git_path)
    dcb.run()


if __name__ == "__main__":
    # import sys
    # if len(sys.argv < 2):
    # raise RuntimeError("Please specify repository path")
    # main(sys.argv[1])
    main("/home/lagunas/devel/external/datasets-sprint/CodeXGLUE/")
