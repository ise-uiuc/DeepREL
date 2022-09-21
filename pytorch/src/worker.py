import matching
from matching import match_api, count_results
import sys
from os.path import join
from utils.loader import load_data
from termcolor import colored


def verify(api_1, api_2, equal_type, root_dir):
    output_dir = join(root_dir, f"{api_1}+{api_2}+rel+{equal_type}")
    neq_dir = join(output_dir, "neq")
    fail_dir = join(output_dir, "fail")
    success_dir = join(output_dir, "success")
    err_dir = join(output_dir, "err")
    bug_dir = join(output_dir, "bug")
    results, indices = match_api(api_1,
                                 api_2,
                                 num=100,
                                 fuzz=False,
                                 equal_type=equal_type,
                                 neq_dir=neq_dir,
                                 fail_dir=fail_dir,
                                 success_dir=success_dir,
                                 err_dir=err_dir,
                                 bug_dir=bug_dir)
    with open(join(root_dir, "log.txt"), "a") as f:
        f.write(f"{api_1} {api_2} {results}\n")
        f.write(str(indices) + "\n")
    return results


def test(api_1, api_2, num, equal_type, root_dir):
    log_file = join(root_dir, f"test-log-{equal_type}.txt")
    verify_file = join(root_dir, f"test-verify-{equal_type}.txt")
    confirmed_file = join(root_dir, f"test-confirmed-{equal_type}.txt")
    candidate_file = join(root_dir, f"test-candidate-{equal_type}.txt")
    error_file = join(root_dir, f"test-error-{equal_type}.txt")

    output_dir = join(root_dir, f"{api_1}+{api_2}+{equal_type}")
    neq_dir = join(output_dir, "neq")
    fail_dir = join(output_dir, "fail")
    success_dir = join(output_dir, "success")
    err_dir = join(root_dir, "error")
    bug_dir = join(root_dir, "bug")

    results, _ = match_api(
        api_1,
        api_2,
        num=num,
        equal_type=equal_type,
        fuzz=True,
        neq_dir=neq_dir,
        fail_dir=fail_dir,
        success_dir=success_dir,
        err_dir=err_dir,
        bug_dir=bug_dir,
    )
    count = {
        "neq": 0,
        "fail": 0,
        "success": 0,
        "err": 0,
        "bug": 0,
    }
    with open(verify_file, "a") as f:
        f.write(f"{api_1} {api_2}\n")

    with open(log_file, "a") as f:
        f.write(f"{api_1} {api_2}\n")
        if results == None:
            f.write("REJECT\n")
        else:
            count = count_results(results)
            # prit(count)
            f.write(str(count) + "\n")
    if count["bug"] > 0:
        with open(confirmed_file, "a") as f_confirm:
            f_confirm.write(f"{api_1} {api_2}\n")
    elif count["err"] > 0:
        with open(error_file, "a") as f_err:
            f_err.write(f"{api_1} {api_2}\n")
    elif count["neq"] > 0:
        with open(candidate_file, "a") as f_candidate:
            f_candidate.write(f"{api_1} {api_2}\n")


if __name__ == "__main__":
    api_1 = sys.argv[1]
    api_2 = sys.argv[2]
    num = int(sys.argv[3])
    root_dir = sys.argv[4]
    matching.root_dir = root_dir

    if verify(api_1, api_2, 1, root_dir):
        # match_value
        print(colored(f"{api_1} {api_2} value match", "green"))
        test(api_1, api_2, num, 1, root_dir)
    elif verify(api_1, api_2, 0, root_dir):
        # match status
        print(colored(f"{api_1} {api_2} status match", "green"))
        test(api_1, api_2, num, 0, root_dir)
    else:
        print(colored(f"{api_1} {api_2} match fail", "red"))