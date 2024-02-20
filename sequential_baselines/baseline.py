import yaml
import os
import numpy as np

# import wandb
import json
from recbole.config import Config
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from logging import getLogger
from recbole.data import create_dataset, data_preparation
from recbole.utils.case_study import full_sort_topk
from evaluation import *
import wandb
import traceback
import torch


def load_config(config_file):
    """Load config and model from config file.

    Args:
        config_file (str): yaml config file path

    Returns:
        model_name (str): model name
        config (Config): config object
    """
    with open(config_file, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    model_name = config["model"]
    config = Config(model=model_name, config_file_list=[config_file])
    return model_name, config

def flatten(matrix):
     flat_list = []
     for row in matrix:
         flat_list.extend(row)
     return flat_list

def ordered_unique(list):
    res = []
    for e in list:
        if e not in res:
            res.append(e)
    return res

def train(config, model_name, train_data, valid_data, test_data, dataset, targets):
    """Train and evaluate the model.

    Args:
        config (Config): config object
        model_name (str): name of the model
        train_data (dataset): train dataset
        valid_data (dataset): valid dataset
        test_data (dataset): test dataset
    """
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    # wandb.init(
    #     project=config["wandb_project"], group=config["wandb_group"], mode="offline"
    # )
    print(f'params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    #print(config["valid_metric"])
    best_valid_score, best_valid_result = trainer.fit(
       train_data, valid_data, verbose=False, show_progress=True   
    )
    
    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]#fist element in array is 'PAD'(default of Recbole) ->remove it 
    #print(external_user_ids)
    topk_items_1by1 = {}
    topk_items = {}

    for internal_user_id in list(range(dataset.user_num))[1:]:
        #print(f'{model.state_parameters}')
        #print(config['topk'])
        _, topk_iid_list = full_sort_topk([internal_user_id], model, test_data, k=config['topk'][0], device=config['device'])
        #print(f'{topk_iid_list=}')
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).tolist()
        #print(f'{external_item_list=}')
        external_item_list = flatten(external_item_list)[:config['topk'][0]]
        #print(f'{external_item_list=}')
        topk_items[external_user_ids[internal_user_id - 1]] = topk_items.get(external_user_ids[internal_user_id - 1], []) + external_item_list

        if not config['next_item']:
            _, topk_iid_list_1by1 = full_sort_topk([internal_user_id], model, test_data, k=1, device=config['device'])
            #print(f'{topk_iid_list=}')
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list_1by1.cpu()).tolist()
            #print(f'{external_item_list=}')
            external_item_list = flatten(external_item_list)
            #print(f'{external_item_list=}')
            topk_items_1by1[external_user_ids[internal_user_id - 1]] = topk_items_1by1.get(external_user_ids[internal_user_id - 1], []) + ordered_unique(external_item_list)
            #print(topk_items_1by1[external_user_ids[internal_user_id - 1]])

    test_result = evaluate(targets, topk_items, 'no_duplicates')
    if not config['next_item']:
        test_result_1by1 = evaluate(targets, topk_items_1by1, 'allow_duplicates_unique')
    else: 
        test_result_1by1 = {
        'precision': 0.0,
        'recall': 0.0,
        'ndcg': 0.0,
        'hit': 0.0
    }
    #print(f'{test_result=}')
    # wandb.finish()

    
    torch.save(model.state_dict(), 'model.pth')
    wandb.save('model.pth')

    return best_valid_score, best_valid_result, test_result, test_result_1by1



def save_results(
    config,
    valid_results,
    test_results,
    valid_name="valid.json",
    test_name="test.json",
    num=None,
):
    """Save the results.

    Args:
        config (Config): config object
        valid_results (dict): valid results
        test_results (dict): test results
        valid_name (str, optional): valid results file name. Defaults to "valid.json".
        test_name (str, optional): test results file name. Defaults to "test.json".
        num (int, optional): number of the experiment. Defaults to None.
    """

    if num is not None:
        results_path = os.path.join(
            config["results_dir"], config["model"], str(num), ""
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    else:
        results_path = os.path.join(config["results_dir"], config["model"])

    with open(os.path.join(results_path, valid_name), "w") as f:
        json.dump(valid_results, f)

    with open(os.path.join(results_path, test_name), "w") as f:
        json.dump(test_results, f)


def update_all_results(
    all_valid_results, all_test_results, valid_results, test_results
):
    """Update all results.

    Args:
        all_valid_results (dict): all valid results
        all_test_results (dict): all test results
        valid_results (dict): valid results
        test_results (dict): test results
    """
    for k, v in valid_results.items():
        all_valid_results[k] = all_valid_results.get(k, []) + [v]

    for k, v in test_results.items():
        all_test_results[k] = all_test_results.get(k, []) + [v]


def mean_and_std(results):
    """Calculate the mean and std of results.

    Args:
        results (dict): results

    Returns:
        dict: mean and std of results
    """
    return {
        k: str(np.char.zfill(str("%.4f" % (100 * np.array(v).mean())), 5))
        + " \\textpm\\ "
        + str("%.2f" % (100 * np.array(v).std()))
        for k, v in results.items()
    }

def get_average_metrics(results):
    """Calculate the mean and std of results.

    Args:
        results (dict): results

    Returns:
        dict: mean and std of results
    """
    return {
        k: {
            'mean': 100 * np.array(v).mean(),
            'std': 100 * np.array(v).std()
        }
        for k, v in results.items()
    }


def print_results(model_name, results, latex_table):
    """Print the results.

    Args:
        results (dict): results
        latex_table (str): results saved in a in latex table format
    """
    latex_row = model_name + " & " + " & ".join(results.values()) + " \\\\\n"
    #print(latex_row)
    with open(latex_table, "a") as f:
        f.write(latex_row)

def parse_targets(file):
    targets = {}
    with open(file, 'r') as f:
        f.readline()
        lines = [line.strip().split('\t') for line in f.readlines()]
    for line in lines:
        targets[line[0]] = targets.get(line[0], []) + [line[1]]
    return targets

def create_row(seed, config, results, message):
    results = [seed, message, config['model'], config['next_item'], results['precision'], results['recall'], results['ndcg'], results['hit']]
    return results

def run(config_file, latex_table, table_results):
    """Run the model.

    Args:
        config_file (str): yaml config file path
        latex_table (str): results saved in a in latex table format
    """

    model_name, config = load_config(config_file)
    wandb.save(config_file)
    init_seed(config["seed"], config["reproducibility"])

    #config['device'] = torch.device('cpu')
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    all_valid_results = dict()
    all_test_results = dict()

    targets = parse_targets(config['test_file'])
    # print(f'{targets=}')

    for num in range(config["run_num"]):
        print("run_num: ", num)
        best_valid_score, best_valid_result, test_result, test_result_1by1 = train(
            config, model_name, train_data, valid_data, test_data, dataset, targets
        )

        table_results.append(create_row(num, config, test_result, 'no_duplicates'))
        
        table_results.append(create_row(num, config, test_result_1by1, 'allow_duplicates'))

        save_results(config, best_valid_result, test_result, num=num)

        update_all_results(
            all_valid_results, all_test_results, best_valid_result, test_result
        )

    save_results(
        config,
        all_valid_results,
        all_test_results,
        valid_name="all_valid.json",
        test_name="all_test.json",
    )

    valid_results = mean_and_std(all_valid_results)
    test_results = mean_and_std(all_test_results)

    save_results(config, valid_results, test_results)

    print_results(model_name, test_results, latex_table)


def run_all(config_file, project_name, run_name):
    """Run all models in the config directory.

    Args:
        config_file (str): yaml config file path
    """

    with open(config_file, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    """if the parent folder of the file config["latex_table"] does not exist, create it"""
    if not os.path.exists(os.path.dirname(config["latex_table"])):
        os.makedirs(os.path.dirname(config["latex_table"]))

    with open(config["latex_table"], "w") as f:
        f.write("\\midrule\n")

    wandb.init(project=project_name, name=run_name)

    table_results = []

    try:
        for config_file in config["config_list"]:
            run(config_file, config["latex_table"], table_results)
    except Exception as error:
        print(error)
         # printing stack trace 
        traceback.print_exc()

    table = wandb.Table(data=table_results, columns=["seed", "message", "model", "next_item", "precision", "recall", "ndcg", "hit"])
    wandb.log({"results": table})

    wandb.finish()
