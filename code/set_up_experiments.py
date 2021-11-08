import json
import os


TRAIN_SCRIPT_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_type gpt2 --tokenizer_name model-configs/{0}-config --config_name model-configs/{0}-config/config.json --train_data_file ../data/wikitext-103-raw/wiki.train.raw --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir train-outputs/{1}/model --do_train --do_eval --evaluate_during_training --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --num_train_epochs 10 --dataloader_drop_last --save_steps 500 --save_total_limit 20 --augmented --augmentation_function {2} --train_function {3} --eval_function {4}"

TRAIN_SCRIPT_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_type gpt2 --tokenizer_name model-configs/{0}-config --config_name model-configs/{0}-config/config.json --train_data_file ../data/wikitext-103-raw/wiki.train.raw --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir train-outputs/{1}/13-model --do_train --do_eval --evaluate_during_training --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --num_train_epochs 10 --dataloader_drop_last --save_steps 500 --save_total_limit 20 --augmented --augmentation_function {2} --train_function {3} --eval_function {4} --seed 13"

TRAIN_SCRIPT_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_type gpt2 --tokenizer_name model-configs/{0}-config --config_name model-configs/{0}-config/config.json --train_data_file ../data/wikitext-103-raw/wiki.train.raw --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir train-outputs/{1}/7-model --do_train --do_eval --evaluate_during_training --per_device_train_batch_size 3 --per_device_eval_batch_size 3 --num_train_epochs 10 --dataloader_drop_last --save_steps 500 --save_total_limit 20 --augmented --augmentation_function {2} --train_function {3} --eval_function {4} --seed 7"

EVAL_SCRIPT_256_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_256_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/13-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/13-{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_256_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/7-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/7-{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FIRST_256_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/{2}-first-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FIRST_256_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/13-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/13-{2}-first-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FIRST_256_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/7-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/7-{2}-first-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FULL_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FULL_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/13-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/13-{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

EVAL_SCRIPT_FULL_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/7-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.valid.raw --output_dir eval-outputs/{1}/7-{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_256_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_256_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/13-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/13-{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_256_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/7-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/7-{2}-256 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_FULL_TEMPLATE = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_FULL_TEMPLATE_13 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/13-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/13-{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"

TEST_SCRIPT_FULL_TEMPLATE_7 = "python transformers/examples/language-modeling/run_language_modeling.py --model_name_or_path train-outputs/{1}/7-model --tokenizer_name model-configs/{0}-config --eval_data_file ../data/wikitext-103-raw/wiki.test.raw --output_dir test-outputs/{1}/7-{2}-1 --do_eval --per_device_eval_batch_size 1 --dataloader_drop_last --augmented --augmentation_function {3} --eval_function {4}"


def set_up_experiments(models):
    if not os.path.isdir("train-scripts"):
        os.mkdir("train-scripts")
    if not os.path.isdir("eval-scripts"):
        os.mkdir("eval-scripts")
    if not os.path.isdir("test-scripts"):
        os.mkdir("test-scripts")
    for model in models:
        make_train_scripts(model, models)
        make_eval_scripts(model, models)
        make_test_scripts(model, models)

def make_train_scripts(model, models):

    if not os.path.isdir("train-scripts/{}".format(model)):
        os.mkdir("train-scripts/{}".format(model))

    if not os.path.exists("train-scripts/{}/train.sh".format(model)):
        train_script = TRAIN_SCRIPT_TEMPLATE.format(models[model]["size"],
                                                    model,
                                                    models[model]["augmentation_function"],
                                                    models[model]["train_function"],
                                                    models[model]["eval_function"])
        with open("train-scripts/{}/train.sh".format(model), "x") as f:
            f.write(train_script)

    if not os.path.exists("train-scripts/{}/train-13.sh".format(model)):
        train_script_13 = TRAIN_SCRIPT_TEMPLATE_13.format(models[model]["size"],
                                                          model,
                                                          models[model]["augmentation_function"],
                                                          models[model]["train_function"],
                                                          models[model]["eval_function"])
        with open("train-scripts/{}/train-13.sh".format(model), "x") as f:
            f.write(train_script_13)

    if not os.path.exists("train-scripts/{}/train-7.sh".format(model)):
        train_script_7 = TRAIN_SCRIPT_TEMPLATE_7.format(models[model]["size"],
                                                        model,
                                                        models[model]["augmentation_function"],
                                                        models[model]["train_function"],
                                                        models[model]["eval_function"])
        with open("train-scripts/{}/train-7.sh".format(model), "x") as f:
            f.write(train_script_7)

def make_eval_scripts(model, models):
    if not os.path.isdir("eval-scripts/{}".format(model)):
        os.mkdir("eval-scripts/{}".format(model))
            
    for other_model in models:
        if models[model]["size"] != models[other_model]["size"]:
            continue

        if not os.path.exists("eval-scripts/{}/{}-256.sh".format(model, other_model)):
            eval_script_256 = EVAL_SCRIPT_256_TEMPLATE.format(models[model]["size"],
                                                              model,
                                                              other_model,
                                                              models[other_model]["augmentation_function_256"],
                                                              models[other_model]["eval_function_256"])
            with open("eval-scripts/{}/{}-256.sh".format(model, other_model), "x") as f:
                f.write(eval_script_256)

        if not os.path.exists("eval-scripts/{}/{}-256-13.sh".format(model, other_model)):
            eval_script_256_13 = EVAL_SCRIPT_256_TEMPLATE_13.format(models[model]["size"],
                                                                    model,
                                                                    other_model,
                                                                    models[other_model]["augmentation_function_256"],
                                                                    models[other_model]["eval_function_256"])
            with open("eval-scripts/{}/{}-256-13.sh".format(model, other_model), "x") as f:
                f.write(eval_script_256_13)

        if not os.path.exists("eval-scripts/{}/{}-256-7.sh".format(model, other_model)):
            eval_script_256_7 = EVAL_SCRIPT_256_TEMPLATE_7.format(models[model]["size"],
                                                                  model,
                                                                  other_model,
                                                                  models[other_model]["augmentation_function_256"],
                                                                  models[other_model]["eval_function_256"])
            with open("eval-scripts/{}/{}-256-7.sh".format(model, other_model), "x") as f:
                f.write(eval_script_256_7)

        if not os.path.exists("eval-scripts/{}/{}-first-256.sh".format(model, other_model)):
            first_256_eval_function = "penultimate_sixth_eval" if models[model]["size"] == 1536 else "penultimate_quarter_eval"
            eval_script_first_256 = EVAL_SCRIPT_FIRST_256_TEMPLATE.format(models[model]["size"],
                                                                          model,
                                                                          other_model,
                                                                          models[other_model]["augmentation_function_256"],
                                                                          first_256_eval_function)
            with open("eval-scripts/{}/{}-first-256.sh".format(model, other_model), "x") as f:
                f.write(eval_script_first_256)

        if not os.path.exists("eval-scripts/{}/{}-first-256-13.sh".format(model, other_model)):
            first_256_eval_function_13 = "penultimate_sixth_eval" if models[model]["size"] == 1536 else "penultimate_quarter_eval"
            eval_script_first_256_13 = EVAL_SCRIPT_FIRST_256_TEMPLATE_13.format(models[model]["size"],
                                                                                model,
                                                                                other_model,
                                                                                models[other_model]["augmentation_function_256"],
                                                                                first_256_eval_function_13)
            with open("eval-scripts/{}/{}-first-256-13.sh".format(model, other_model), "x") as f:
                f.write(eval_script_first_256_13)

        if not os.path.exists("eval-scripts/{}/{}-first-256-7.sh".format(model, other_model)):
            first_256_eval_function_7 = "penultimate_sixth_eval" if models[model]["size"] == 1536 else "penultimate_quarter_eval"
            eval_script_first_256_7 = EVAL_SCRIPT_FIRST_256_TEMPLATE_7.format(models[model]["size"],
                                                                              model,
                                                                              other_model,
                                                                              models[other_model]["augmentation_function_256"],
                                                                              first_256_eval_function_7)
            with open("eval-scripts/{}/{}-first-256-7.sh".format(model, other_model), "x") as f:
                f.write(eval_script_first_256_7)

        if not os.path.exists("eval-scripts/{}/{}-1.sh".format(model, other_model)):
            eval_script_full = EVAL_SCRIPT_FULL_TEMPLATE.format(models[model]["size"],
                                                                model,
                                                                other_model,
                                                                models[other_model]["augmentation_function_full"],
                                                                models[other_model]["eval_function_full"])
            with open("eval-scripts/{}/{}-1.sh".format(model, other_model), "x") as f:
                f.write(eval_script_full)

        if not os.path.exists("eval-scripts/{}/{}-1-13.sh".format(model, other_model)):
            eval_script_full_13 = EVAL_SCRIPT_FULL_TEMPLATE_13.format(models[model]["size"],
                                                                      model,
                                                                      other_model,
                                                                      models[other_model]["augmentation_function_full"],
                                                                      models[other_model]["eval_function_full"])
            with open("eval-scripts/{}/{}-1-13.sh".format(model, other_model), "x") as f:
                f.write(eval_script_full_13)

        if not os.path.exists("eval-scripts/{}/{}-1-7.sh".format(model, other_model)):
            eval_script_full_7 = EVAL_SCRIPT_FULL_TEMPLATE_7.format(models[model]["size"],
                                                                    model,
                                                                    other_model,
                                                                    models[other_model]["augmentation_function_full"],
                                                                    models[other_model]["eval_function_full"])
            with open("eval-scripts/{}/{}-1-7.sh".format(model, other_model), "x") as f:
                f.write(eval_script_full_7)

def make_test_scripts(model, models):
    if not os.path.isdir("test-scripts/{}".format(model)):
        os.mkdir("test-scripts/{}".format(model))
            
    for other_model in models:
        if models[model]["size"] != models[other_model]["size"]:
            continue

        if not os.path.exists("test-scripts/{}/{}-256.sh".format(model, other_model)):
            test_script_256 = TEST_SCRIPT_256_TEMPLATE.format(models[model]["size"],
                                                              model,
                                                              other_model,
                                                              models[other_model]["augmentation_function_256"],
                                                              models[other_model]["eval_function_256"])
            with open("test-scripts/{}/{}-256.sh".format(model, other_model), "x") as f:
                f.write(test_script_256)

        if not os.path.exists("test-scripts/{}/{}-256-13.sh".format(model, other_model)):
            test_script_256_13 = TEST_SCRIPT_256_TEMPLATE_13.format(models[model]["size"],
                                                                    model,
                                                                    other_model,
                                                                    models[other_model]["augmentation_function_256"],
                                                                    models[other_model]["eval_function_256"])
            with open("test-scripts/{}/{}-256-13.sh".format(model, other_model), "x") as f:
                f.write(test_script_256_13)

        if not os.path.exists("test-scripts/{}/{}-256-7.sh".format(model, other_model)):
            test_script_256_7 = TEST_SCRIPT_256_TEMPLATE_7.format(models[model]["size"],
                                                                  model,
                                                                  other_model,
                                                                  models[other_model]["augmentation_function_256"],
                                                                  models[other_model]["eval_function_256"])
            with open("test-scripts/{}/{}-256-7.sh".format(model, other_model), "x") as f:
                f.write(test_script_256_7)

        if not os.path.exists("test-scripts/{}/{}-1.sh".format(model, other_model)):
            test_script_full = TEST_SCRIPT_FULL_TEMPLATE.format(models[model]["size"],
                                                                model,
                                                                other_model,
                                                                models[other_model]["augmentation_function_full"],
                                                                models[other_model]["eval_function_full"])
            with open("test-scripts/{}/{}-1.sh".format(model, other_model), "x") as f:
                f.write(test_script_full)

        if not os.path.exists("test-scripts/{}/{}-1-13.sh".format(model, other_model)):
            test_script_full_13 = TEST_SCRIPT_FULL_TEMPLATE_13.format(models[model]["size"],
                                                                      model,
                                                                      other_model,
                                                                      models[other_model]["augmentation_function_full"],
                                                                      models[other_model]["eval_function_full"])
            with open("test-scripts/{}/{}-1-13.sh".format(model, other_model), "x") as f:
                f.write(test_script_full_13)

        if not os.path.exists("test-scripts/{}/{}-1-7.sh".format(model, other_model)):
            test_script_full_7 = TEST_SCRIPT_FULL_TEMPLATE_7.format(models[model]["size"],
                                                                    model,
                                                                    other_model,
                                                                    models[other_model]["augmentation_function_full"],
                                                                    models[other_model]["eval_function_full"])
            with open("test-scripts/{}/{}-1-7.sh".format(model, other_model), "x") as f:
                f.write(test_script_full_7)


if __name__ == '__main__':

    with open("models.txt", "r") as f:
        models_list = f.readlines()

    models = {}

    for m in models_list[1:]:
        model, a_f, t_f, e_f = map(lambda x: x.strip(), m.split(","))
        size = sum(int(x) for x in model.split("-")[0].split("+"))
        if size == 1024:
            # if "identity" in a_f:
            #     a_f_256 = "identity_quarter"
            # else:
            if "identity" in a_f and "old" in a_f:
                a_f_256 = "identity_old_quarter"
            else:
                a_f_256 = a_f + "_quarter"
            e_f_256 = "last_quarter_eval"
        elif size == 1536:
            if "identity_third" == a_f:
                a_f_256 = "identity_sixth"
            else:
                a_f_256 = a_f + "_sixth"
            e_f_256 = "last_sixth_eval"
        else:
            raise ValueError("Invalid model size {}".format(size))
        if "identity" in a_f and "old" not in a_f:
            a_f_full = "identity_full"
        else:
            a_f_full = a_f + "_full"
        e_f_full = "last_element_eval"
        models[model] = {"size": size,
                         "augmentation_function": a_f,
                         "train_function": t_f,
                         "eval_function": e_f,
                         "augmentation_function_256": a_f_256,
                         "eval_function_256": e_f_256,
                         "augmentation_function_full": a_f_full,
                         "eval_function_full": e_f_full}

    set_up_experiments(models)
