import os

yaml_collection_path = "stackex-votes_model.channels_64_128_model.num_layers_2_3_model.use_self_join_True_False_optim.base_lr_0.01_0.001"

base_dir = "results"
seed = 42

runs = [name[:-5] for name in os.listdir(os.path.join(base_dir, yaml_collection_path))]

search_string = "Test metrics: "
for run in runs:
    print("*" * 8)
    cur_path = os.path.join(base_dir, run, str(seed), "logging.log")
    test_stats = None
    try:
        with open(cur_path, "r") as f:
            for line_number, line in enumerate(f, 1):  # Starting at line 1
                # Check if the search string is in the current line
                if search_string in line:
                    # Return the line along with its line number
                    test_stats = line[len(search_string) :]
                    break
    except:
        print(f"{cur_path} does not have a log")
    if test_stats is None:
        print(f"{run} does not have test stats")
        continue
    test_stats = eval(test_stats)
    print(run)
    print(test_stats)
    print("*" * 8)
    # test_stats.replace('\'', '"')
    # test_stats = json.loads(test_stats)
    # print(test_stats)
