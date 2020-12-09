import json
from pathlib import Path
import sh

DATASET_PATH = None


def test_base(name, lens = {}, config = "default"):
    from datasets import load_dataset
    file_path = DATASET_PATH  /name
    dataset = load_dataset(path=str(file_path), name = config)
    sizes[name] = {}
    for split, split_len in lens.items():
        l = len(dataset[split])
        if l != split_len:
            if split_len is None:
                print(f"Got dataset length name={name}, split={split} : len = {l}")
            else:
                raise RuntimeError(f"Invalid dataset length name={name}, split={split}: {l} != {split_len}(reference)")

if False:
    def test_cc_code_to_code_trans():
        test_base("cc_code_to_code_trans", {"train": 10300, "validation": 500, "test": 1000})

    def test_cc_defect():
        test_base("cc_defect_detection", {"train":21854, "validation":2732, "test":2732})

    def test_cc_clone_detection_big_clone_bench():
        test_base("cc_clone_detection_big_clone_bench", {"train":901028, "validation":415416, "test":415416})

    def test_cc_clone_detection_poj_104():
        test_base("cc_clone_detection_poj_104", {"train":32000, "validation":8000, "test":12000})

    def test_cc_cloze_testing_all():
        counts = {"ruby": 4437, "javascript": 13837, "go": 25282, "python": 40137, "java": 40492, "php": 51930}

        for language, count in counts.items():
            test_base("cc_cloze_testing_all", dict(train=count), config=language)

    def test_cc_cloze_testing_maxmin():
        counts = {"ruby": 38, "javascript": 272, "go": 152, "python": 1264, "java": 482, "php": 407}

        for language, count in counts.items():
            test_base("cc_cloze_testing_maxmin", dict(train=count), config=language)

    def test_cc_code_completion_line():
        counts = {"java": 3000, "python":10000}

        for language, count in counts.items():
            test_base("cc_code_completion_line", dict(train=count), config=language)

    def test_cc_code_completion_token():
        counts = {"java": {"train":12934, "validation":7189, "test":8268}, "python": {"train":100000, "test":50000}}

        for language, count_dict in counts.items():
            test_base("cc_code_completion_token", count_dict, config=language)

    def test_cc_code_refinement():
        counts = {"small": {"train":46680, "validation":5835, "test":5835}, "medium": {"train":52364, "validation":6546, "test":6545}}
        for size in "small", "medium":
            test_base("cc_code_refinement", counts[size], config=size)


    def test_ct_code_to_text():
        language_counts = dict(
            python=[251820, 13914, 14918],
            php=[241241, 12982, 14014],
            go=[167288, 7325, 8122],
            java=[164923, 5183, 10955],
            javascript=[58025, 3885, 3291],
            ruby=[24927, 1400, 1261],
        )

        for language, counts in language_counts.items():
            test_base("ct_code_to_text", {"train":counts[0], "validation":counts[1], "test":counts[2]}, config = language)

    def test_tt_text_to_text():
        language_pairs_count = dict(da_en=[42701, 1000, 1000], lv_en=[18749, 1000, 1000], no_en=[44322,1000,1000], zh_en=[50154,1000,1000])

        for language, counts in language_pairs_count.items():
            test_base("tt_text_to_text", {"train":counts[0], "validation":counts[1], "test":counts[2]}, config = language)

    def test_tc_text_to_code():
        counts = {"train":100000, "validation": 2000, "test": 2000}
        test_base("tc_text_to_code", counts)

    def test_tc_nl_code_search_adv():
        counts = {"train":251820, "validation": 9604, "test": 19210}
        test_base("tc_nl_code_search_adv", counts)

    def test_tc_nl_code_search_web_query():
        counts = {"train":251820, "validation": 9604, "test": 19210}
        test_base("tc_nl_code_search_web_query", counts)

def run_own_tests(dataset_name):
    dataset_size = sizes[dataset_name[len("code_x_glue_"):]]
    for config_name, split_sizes in dataset_size.items():
        #print("TESTING", config_name, split_sizes)
        try:
            test_base(dataset_name, lens=split_sizes, config=config_name)
        except:
            print(f"ERROR with {dataset_name}.{config_name}")
            raise

_fg = True

def create_dummy_data(datasets_path, dataset_name):
    dataset_path = datasets_path / dataset_name
    full_path = str(dataset_path.resolve())
    sh.datasets_cli("dummy_data", full_path, "--auto_generate", "--keep_uncompressed", _fg=_fg)

import contextlib
import os

@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)


def run_datasets_tests(datasets_path, dataset_name, dummy = False):
    test_name = "dataset_all_configs" if dummy else "real_dataset"

    with cd(datasets_path.parent):
        sh.pytest(f"tests/test_dataset_common.py::LocalDatasetTest::test_load_{test_name}_{dataset_name}", _fg = _fg, _env={"RUN_SLOW":"1"})

def run_dataset_info_create(datasets_path, dataset_name):
    with cd(datasets_path.parent):
        sh.datasets_cli("test", f"datasets/{dataset_name}", "--save_infos", "--all_configs", _fg=_fg)

def cleanup_code(datasets_path, dataset_name):
    for name in ["common.py", dataset_name + ".py"]:
        full_path = datasets_path/dataset_name/name
        #print("RUNNING autoflake", full_path)
        sh.autoflake("--in-place", "--remove-all-unused-imports", str(full_path), _fg=_fg)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise RuntimeError(
            "you should specify a single argument: path to the local version of https://github.com/huggingface/datasets repository")

    DATASET_PATH = Path(sys.argv[1]) / "datasets"

    current_dir = Path(__file__).parent
    sizes = json.loads((current_dir / "sizes.json").open().read())

    dirs = []
    for dir in DATASET_PATH.iterdir():
        if dir.name.startswith("code_x_glue_"):
            dirs += [dir]

    for dir in dirs:
        dataset_name = dir.name
        cleanup_code(DATASET_PATH, dataset_name)

    issues = []
    for dir in dirs:
        dataset_name = dir.name
        #if dataset_name != "code_x_glue_cc_clone_detection_poj_104":
        #    continue

        run_own_tests(dataset_name)

        run_datasets_tests(DATASET_PATH, dataset_name, dummy=False)

        try:
            #create_dummy_data(DATASET_PATH, dataset_name)
            run_datasets_tests(DATASET_PATH, dataset_name, dummy=True)
        except Exception as e:
            raise
            issues += [dataset_name]
            print("ERROR", dataset_name, e)

        run_dataset_info_create(DATASET_PATH, dataset_name)


    print("ISSUES:")
    for k in issues:
        print(k)



