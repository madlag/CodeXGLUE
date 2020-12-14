import contextlib
import os
import sh
import os.path

renames = {"cc_defect_detection":"cc_defect_detect",
"cc_clone_detection_big_clone_bench":"cc_clone_detect_big",
"cc_code_refinement":"cc_refine",
"cc_code_completion_token":"cc_complete_token",
"cc_code_to_code_trans":"cc_code_to_code",
"cc_code_completion_line":"cc_code_complete_line",
"cc_clone_detection_poj_104":"cc_clone_detect_poj",
"cc_cloze_testing_maxmin":"cc_cloze_test_maxmin",
"cc_cloze_testing_all":"cc_cloze_test_all",
"ct_code_to_text":"ct_code_to_text",
"tt_text_to_text":"tt_text_to_text",
#"tc_nl_code_search_web_query": "tc_search_web_query",
"tc_text_to_code":"tc_text_to_code",
"tc_nl_code_search_adv":"tc_search_adv"}



@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)


base_path = "/home/lagunas/devel/hf/datasets/datasets"
with cd(base_path):
    for k,v in renames.items():
        src = "code_x_glue_" + k
        dest = "cxg" + v
        assert(os.path.exists(src))
        #sh.git("move", src, dest)