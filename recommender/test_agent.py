from __future__ import absolute_import, division, print_function
import json
from collections import Counter
import os
import argparse
from math import log
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
from functools import reduce
from kg_env import BatchKGEnvironment
from actor_critic import ActorCritic, RNN, Embeds, Encoder
from utils import *
#import wandb

#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def evaluate(topk_matches, test_user_products, use_wandb, tmp_dir, exp_name, result_file_name='result.txt', min_courses=10):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    #print(topk_matches, test_user_products)
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits, hits_at_1, hits_at_5, hits_at_10 = [], [], [], [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        is_invalid = False
        if uid not in topk_matches or len(topk_matches[uid]) < min_courses: 
            invalid_users.append(uid)
            is_invalid = True
        pred_list, rel_set = topk_matches.get(uid, []), test_user_products[uid]
        if is_invalid == False:
            hit_num = 0.0
            dcg = 0.0
            #print(pred_list, rel_set, end='')
            for i in range(len(pred_list)):
                if pred_list[i] in rel_set:
                    dcg += 1. / (log(i + 2) / log(2))
                    hit_num += 1

            # idcg
            idcg = 0.0
            for i in range(min(len(rel_set), len(pred_list))):
                idcg += 1. / (log(i + 2) / log(2))
            hit = 1.0 if hit_num > 0.0 else 0.0
            ndcg = dcg / idcg
            recall = hit / len(rel_set)
            precision = hit / len(pred_list)
            ndcgs.append(ndcg)
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)
        
    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100

    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}\n'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))

    filename = tmp_dir + f"/evaluation/{exp_name}/" + result_file_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}\n'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
        f.write('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}\n'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
    if use_wandb:
        wandb.save(filename)
    
    return avg_precision, avg_recall, avg_ndcg, avg_hit


def evaluate_validation(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    hits= []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                hit_num += 1

        hit = 1.0 if hit_num > 0.0 else 0.0
        hits.append(hit)

    avg_hit = np.mean(hits) * 100

    print(' HR={:.3f} | Invalid users={}\n'.format(avg_hit, len(invalid_users)))    
    
    return avg_hit


#def batch_beam_search(env, model, hist_encoder, kg_args, uids, device, topk=[10, 3, 1], policy=0, seq_batch_size=3):
#    def _batch_acts_to_masks(batch_acts):
#        batch_masks = []
#        for acts in batch_acts:
#            num_acts = len(acts)
#            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
#            act_mask[:num_acts] = 1
#            batch_masks.append(act_mask)
#        return np.vstack(batch_masks)
#    
#    _, hist_state = hist_encoder(uids)
#
#    path_pool = env._batch_path  # list of list, size=bs
#    probs_pool = [[] for _ in uids]
#    model.eval()
#    predictions = [[] for _ in uids]
#    probabilities = [[] for _ in uids]
#
#    env.reset_predictions(len(uids))
#    for hop in range(50): 
#        
#        ### Start batch episodes ###
#        state_pool = env.reset(uids, hist_state.squeeze())  # numpy of [bs, dim]
#        path_pool = env._batch_path  # list of list, size=bs
#        probs_pool = [[] for _ in uids]
#        for i in range(seq_batch_size):
#            state_tensor = torch.FloatTensor(state_pool).to(device)
#            acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
#            actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
#            actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
#            batch_act_embeddings = env.batch_action_embeddings(path_pool, acts_pool)  # numpy array of size [bs, 2*embed_size, act_dim]
#            embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
#            probs, _ = model((state_tensor, actmask_tensor, embeddings))  # Tensor of [bs, act_dim]
#            probs = probs + actmask_tensor.float()  # In order to differ from masked actions
#            topk_probs, topk_idxs = torch.topk(probs, 1, dim=1)  # LongTensor of [bs, k]
#            topk_idxs = topk_idxs.detach().cpu().numpy()
#            topk_probs = topk_probs.detach().cpu().numpy()
#            
#            new_path_pool, new_probs_pool = [], []
#            for row in range(topk_idxs.shape[0]):
#                path = path_pool[row]
#                probs = probs_pool[row]
#                for idx, p in zip(topk_idxs[row], topk_probs[row]):
#                    if idx >= len(acts_pool[row]):  # act idx is invalid
#                        continue
#                    relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
#                    if relation == kg_args.self_loop:
#                        next_node_type = path[-1][1]
#                    else:
#                        next_node_type = kg_args.kg_relation[path[-1][1]][relation]
#                    new_path = path + [(relation, next_node_type, next_node_id)]
#                    new_path_pool.append(new_path)
#                    new_probs_pool.append(probs + [p])
#            path_pool = new_path_pool
#            probs_pool = new_probs_pool
#            if hop < len(topk) - 1:  # no need to update state at the last hop
#                state_pool = env._batch_get_state(path_pool, predictions)
#        
#        for i, path_probs in enumerate(zip(path_pool, probs_pool)):
#            _, node_type, node_id = path_probs[0][-1]
#            probs = sum(path_probs[1])
#            if(node_type == 'item'):
#                predictions[i].append(node_id)
#                probabilities[i].append(probs)
#        
#            ### End of episodes ###
#        
#    return predictions, probabilities

def batch_beam_search(env, model, hist_encoder, kg_args, uids, device, topk=[10, 3, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    
    _, hist_state = hist_encoder(uids)
    env.reset_predictions(len(uids))
    state_pool = env.reset(uids, hist_state.squeeze())  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    hist_encoder.eval()
    
    for hop in range(len(topk)): 
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        batch_act_embeddings = env.batch_action_embeddings(path_pool, acts_pool)  # numpy array of size [bs, 2*embed_size, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
        probs, _ = model((state_tensor, actmask_tensor, embeddings))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == kg_args.self_loop:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_args.kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < len(topk) - 1:  # no need to update state at the last hop
            state_pool = env._batch_get_state(path_pool, [])
    #print(path_pool, probs_pool)
    return path_pool, probs_pool

def predict_paths(policy_file, encoder_file, path_file, args, kg_args, data='test_target'):
    print('Predicting paths...')
    hist_encoder = RNN(args.tmp_dir, args.embedding_size, args.embedding_size, args.history_dim, labels='test').to(args.device)
    env = BatchKGEnvironment(args.tmp_dir, kg_args, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history, history_dim=args.history_dim, reward_function=args.reward, use_pattern=args.use_pattern, use_enroll=args.use_enroll)
    pretrain_model_sd = torch.load(policy_file, map_location=torch.device('cpu'))
    pretrain_encoder_sd = torch.load(encoder_file, map_location=torch.device('cpu'))
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden, modified_policy=args.modified_policy, embed_size=env.embed_size).to(args.device)
    
    model_sd = model.state_dict()
    model_sd.update(pretrain_model_sd)
    model.load_state_dict(model_sd)

    encoder_sd = hist_encoder.state_dict()
    encoder_sd.update(pretrain_encoder_sd)
    hist_encoder.load_state_dict(encoder_sd)

    test_labels = load_labels(args.tmp_dir, data)
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    pbar = tqdm(total=len(test_uids))

    res = []
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, hist_encoder, kg_args, batch_uids, args.device, topk=args.topk)
        for tuple in zip(paths, probs):
            uid = tuple[0][0][2]
            res.append((uid, tuple[0], tuple[1]))
        start_idx = end_idx
        pbar.update(batch_size)
    pickle.dump(res, open(path_file, 'wb'))
    if args.use_wandb:
        wandb.save(path_file)


# def evaluate_paths(dir_path, path_file, train_labels, test_labels, kg_args, use_wandb, validation=False, sum_prob=False):
#     embeds = load_embed(dir_path)
#     user_embeds = embeds['user']
#     enroll_embeds = embeds[kg_args.interaction][0]
#     course_embeds = embeds['item']
#     scores = np.dot(user_embeds + enroll_embeds, course_embeds.T)
# 
#     # 1) Get all valid paths for each user, compute path score and path probability.
#     results = pickle.load(open(path_file, 'rb'))
#     pred_paths = {uid: [] for uid in test_labels}
#     for uid, path, probs in results:
#         probs = torch.tensor(probs)
#         topk_probs, topk_idxs = torch.topk(probs, 20)
#         path = torch.tensor(path)[torch.sort(topk_idxs).values]
#         
#         pred_paths[uid] = list(path)
#         
#     if validation == True:
#         return evaluate_validation(pred_paths, test_labels, use_wandb)
#     else:  
#         evaluate(pred_paths, test_labels, use_wandb, args.tmp_dir, args.exp_dir, result_file_name=f"result.txt", min_courses=20)

def evaluate_paths(path_file, test_target_labels, test_labels, use_wandb, validation=False, sum_prob=False, next_item=True):

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: [] for uid in test_target_labels}
    #print(results)
    for uid, path, probs in results:
        uid = path[0][-1]
        path = list(filter(lambda p: True if p[0] != 'self_loop' else False, path))
        pred = []
        #path_score = 0
        for node in path:
            _, type, eid = node
            if(type == 'item'):
                pred.append(eid)
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid].append((path_prob, path, pred[1:]))
    best_pred_paths = {}
    for uid in pred_paths:
        best_pred_paths[uid] = []
        # Get the paths with highest probability
        sorted_path = sorted(pred_paths[uid], key=lambda x: x[0], reverse=True)
        best_pred_paths[uid]=sorted_path

    pred_labels = {}
    for uid in best_pred_paths:
        top_seqs = [p for _, _, p in best_pred_paths[uid]]  # from largest to smallest
        pred_labels[uid] = top_seqs 

    if next_item:
        labels = {}
        # get top 10 items for evaluation
        for x in pred_labels:
            items = [item for items in pred_labels[x] for item in items if item not in test_labels[x]]
            items = list(dict.fromkeys(items))[:10]
            labels[x] = items
        pred_labels = labels
        min_threshold = 10
    else:
        # get top 1 sequence of items (len of sequence = batch size)
        pred_labels = {x: pred_labels[x][0] if len(pred_labels[x]) > 0 else [] for x in pred_labels}
        min_threshold = 1

    with open('./preds_test_labels.pkl', 'wb') as f:
        pickle.dump([pred_labels, test_target_labels], f)

    if validation == True:
        return evaluate_validation(pred_paths, test_target_labels, use_wandb)
    else:  
        return evaluate(pred_labels, test_target_labels, use_wandb, args.tmp_dir, args.exp_dir, result_file_name=f"result.txt", min_courses=min_threshold)


def update_kg(args, train_target_labels):
    kg = load_kg(args.tmp_dir)

    for user in train_target_labels:
        kg.add_train_targets(user, train_target_labels[user])

    save_kg(args.tmp_dir, kg)

def test(args, kg_args):
    policy_file = args.log_dir + '/tmp_policy_model_epoch_{}.ckpt'.format(args.epochs)
    encoder_file = args.log_dir + '/hist_encoder_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch_{}.pkl'.format(args.epochs)

    train_target_labels = load_labels(args.tmp_dir, 'train_target')
    test_labels = load_labels(args.tmp_dir, 'test')
    test_target_labels = load_labels(args.tmp_dir, 'test_target')

    update_kg(args, train_target_labels)

    if args.run_path:
        predict_paths(policy_file, encoder_file, path_file, args, kg_args)
    if args.run_eval:
        avg_precision, avg_recall, avg_ndcg, avg_hit = evaluate_paths(path_file, test_target_labels, test_labels, args.use_wandb, args.result_file_name, args.sum_prob, args.next_item)
        if args.use_wandb:
            results = [wandb.run.name, args.seed, args.reward, args.next_item, args.state_history, args.history_dim, args.max_path_len, args.use_enroll, avg_precision, avg_recall, avg_ndcg, avg_hit]
            table = wandb.Table(data=[results], columns=["run_name", "seed", "reward", "next_item", "state_history", "history_dim","max_path_len", "use_enroll", "precision", "recall", "ndcg", "hit"])
            wandb.log({"results": table})

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/mooc_with_enroll/3.json", help="Config file."
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = edict(json.load(f))

    args = config.TEST_AGENT
    assert(args.max_path_len == len(args.topk))

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cpu') if torch.cuda.is_available() else 'cpu'

    print(args.device)
    if(args.early_stopping == True):
        with open("early_stopping.txt", 'r') as f:
            args.epochs = int(f.read())

    args.log_dir = '{}/{}/{}'.format(args.tmp_dir, args.exp_dir, args.name)
    set_random_seed(args.seed)
    test(args, config.KG_ARGS)
    filename = args.tmp_dir + "/evaluation/" + args.exp_dir + "/" + f'{args.result_file_name}'

    with open(filename, "w") as f:
        f.write(f'reward={args.reward} |  pattern={args.use_pattern} | modified_policy={args.modified_policy} \n')

    if args.use_wandb:
        wandb.save('./preds_test_labels.pkl')
        wandb.finish()

