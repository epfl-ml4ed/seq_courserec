from math import log
import numpy as np
import pickle
import wandb

def evaluate(targets, predictions, message):
    #print(targets, predictions)
    with open(f'res_{message}.pkl', 'wb') as f:
        pickle.dump({'tgt': targets, 'preds': predictions}, f)
    wandb.save(f'res_{message}.pkl')

    invalid_users = []
    precisions, recalls, ndcgs, hits, hits_at_1, hits_at_5, hits_at_10 = [], [], [], [], [], [], []
    test_user_idxs = list(targets.keys())
    for uid in test_user_idxs:
        is_invalid = False
        if uid not in predictions or len(predictions[uid]) < 1:
            invalid_users.append(uid)
            is_invalid = True
        pred_list, rel_set = predictions.get(uid, []), targets[uid]
        if is_invalid == False:
            hit_num = 0.0
            dcg = 0.0
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

        
    avg_precision = np.mean(precisions) 
    avg_recall = np.mean(recalls) 
    avg_ndcg = np.mean(ndcgs) 
    avg_hit = np.mean(hits) 

    result = {
        'precision': avg_precision,
        'recall': avg_recall,
        'ndcg': avg_ndcg,
        'hit': avg_hit
    }

    return result


