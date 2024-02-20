from __future__ import absolute_import, division, print_function

import os
import argparse
import random
import wandb
import pickle
import random
from sklearn.model_selection import train_test_split
import json

from utils import *
from data_utils import Dataset
from knowledge_graph import KnowledgeGraph
from easydict import EasyDict as edict


def generate_labels(data_dir, mode="train"):
    enrolment_file = "{}/{}.txt".format(data_dir, mode)
    user_courses = {}  # {uid: [cid,...], ...}
    with open(enrolment_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            arr = line.split(" ")
            user_idx = int(arr[0])
            course_idx = int(arr[1])
            if user_idx not in user_courses:
                user_courses[user_idx] = []
            user_courses[user_idx].append(course_idx)
    return user_courses


def split_train_test_data(
    data_dir, ratio=0.9, ratio_validation=0.05, data_file="enrolments.txt"
):
    path = data_dir + "/" + data_file
    with open(path, encoding="utf-8") as f:
        data = f.readlines()
    train_data, validation_test_data = train_test_split(data, train_size=ratio)
    validation_data, test_data = train_test_split(
        validation_test_data, train_size=ratio_validation
    )
    create_data_file(data_dir, train_data, "train.txt")
    create_data_file(data_dir, validation_data, "validation.txt")
    create_data_file(data_dir, test_data, "test.txt")


def split_train_test_data_by_user(
    data_dir, ratio=0.8, ratio_validation=0.5, data_file="enrolments_by_user.pkl", seq_batch_size=1
):
    path = data_dir + "/" + data_file
    with open(path, "rb") as f:
        learner_courses = pickle.load(f)
    train_data = []
    test_data = []
    train_targets = []
    test_targets = []

    for learner in learner_courses:
        courses = learner_courses[learner]
        l_train_data = courses[:-seq_batch_size * 2]
        l_train_target = courses[-seq_batch_size*2:-seq_batch_size]
        l_test_data = courses[:-seq_batch_size]
        l_test_target = courses[-seq_batch_size:]

        for c in l_train_data:
            train_data.append(f"{learner} {c}\n")
        for c in l_test_data:
            test_data.append(f"{learner} {c}\n")
        for c in l_train_target:
            train_targets.append(f"{learner} {c}\n")
        for c in l_test_target:
            test_targets.append(f"{learner} {c}\n")   

    # random.shuffle(train_data)
    # random.shuffle(test_data)
    # random.shuffle(validation_data)

    create_data_file(data_dir, train_data, "train.txt")
    create_data_file(data_dir, train_targets, "train_target.txt")
    create_data_file(data_dir, test_targets, "test_target.txt")
    create_data_file(data_dir, test_data, "test.txt")

def fixed_split_train_test_data_by_user(
    data_dir, num_train=2, num_val=4, num_test=6, data_file="enrolments_by_user.pkl"
):
    path = data_dir + "/" + data_file
    with open(path, "rb") as f:
        learner_courses = pickle.load(f)
    train_data = []
    test_data = []
    train_targets = []
    test_targets = []

    for learner in learner_courses:
        courses = learner_courses[learner]
        l_train_data = courses[:num_train]
        l_train_target = courses[num_train:num_train + num_val]
        l_test_data = courses[:num_train + num_val]
        l_test_target = courses[-num_test:]

        for c in l_train_data:
            train_data.append(f"{learner} {c}\n")
        for c in l_test_data:
            test_data.append(f"{learner} {c}\n")
        for c in l_train_target:
            train_targets.append(f"{learner} {c}\n")
        for c in l_test_target:
            test_targets.append(f"{learner} {c}\n")   

    create_data_file(data_dir, train_data, "train.txt")
    create_data_file(data_dir, train_targets, "train_target.txt")
    create_data_file(data_dir, test_targets, "test_target.txt")
    create_data_file(data_dir, test_data, "test.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config_mooc.json", help="Config file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.PREPROCESS

    set_random_seed(args.seed)
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=config.PREPROCESS)

    # Create train and test data
    # split_train_test_data(args.dataset, ratio=args.ratio, ratio_validation=args.ratio_validation, data_file=args.data_file)
    fixed_split_train_test_data_by_user(
        args.data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        data_file=args.data_file,
    )
    # Create MoocDataset instance for dataset.
    # ========== BEGIN ========== #
    print(f"Loading dataset from folder: {args.data_dir}")
    if not os.path.isdir(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    dataset = Dataset(args.data_dir, config.KG_ARGS)
    save_dataset(args.tmp_dir, dataset, args.use_wandb)

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print("Creating knowledge graph from dataset...")
    dataset = load_dataset(args.tmp_dir)
    kg = KnowledgeGraph(dataset, config.KG_ARGS, use_user_relations=args.use_user_relations, use_entity_relations=args.use_entity_relations)
    kg.compute_degrees()
    save_kg(args.tmp_dir, kg, args.use_wandb)
    # =========== END =========== #

    # Genereate train/test labels.
    # ========== BEGIN ========== #
    print("Generate train/test labels.")
    train_labels = generate_labels(args.data_dir, "train")
    test_labels = generate_labels(args.data_dir, "test")
    train_target_labels = generate_labels(args.data_dir, "train_target")
    test_target_labels = generate_labels(args.data_dir, "test_target")
    
    save_labels(args.tmp_dir, train_labels, mode="train", use_wandb=args.use_wandb)
    save_labels(args.tmp_dir, test_labels, mode="test", use_wandb=args.use_wandb)
    save_labels(args.tmp_dir, train_target_labels, mode="train_target", use_wandb=args.use_wandb)
    save_labels(args.tmp_dir, test_target_labels, mode="test_target", use_wandb=args.use_wandb)

    # =========== END =========== #
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
