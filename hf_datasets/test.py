import json
sizes = {}

tests = {}
def test_base(name, lens = {}, config = "default"):
    print("TESTING", name, config)
    if name not in tests:
        tests[name] = {}
    tests[name][config] = lens

    name = "code_x_glue_" + name

    from datasets import load_dataset
    file_path = f"/home/lagunas/devel/hf/datasets/datasets/{name}"
    print(f"file_path={file_path}, config={config}")
    dataset = load_dataset(path=file_path, name = config)
    sizes[name] = {}
    for split, split_len in lens.items():
        l = len(dataset[split])
        if l != split_len:
            if split_len is None:
                print(f"Got dataset length name={name}, split={split} : len = {l}")
            else:
                raise RuntimeError(f"Invalid dataset length name={name}, split={split}: {l} != {split_len}(reference)")

        sizes[name][split] = l

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

if __name__ == "__main__":
    test_tt_text_to_text()

    if True:
        test_ct_code_to_text()
        test_ct_code_to_text()

        test_tc_text_to_code()
        test_cc_code_refinement()
        test_cc_code_completion_token()
        test_cc_code_completion_line()
        test_cc_cloze_testing_maxmin()
        test_cc_cloze_testing_all()
        test_cc_clone_detection_poj_104()
        test_cc_clone_detection_big_clone_bench()
        test_cc_code_to_code_trans()
        test_cc_defect()
        #test_tc_nl_code_search_web_query()
        test_tc_nl_code_search_adv()

    #with open("test_generated.json", "w") as f:
    #    f.write(json.dumps(tests, indent=4, sort_keys=True))

