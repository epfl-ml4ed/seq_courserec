from __future__ import absolute_import, division, print_function

import random
from utils import *
from torchtext.data.metrics import bleu_score


class KGState(object):
    def __init__(self, embed_size, history_len=1, history_dim=32):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size + history_dim
        elif history_len == 1:
            self.dim = 4 * embed_size + history_dim
        elif history_len == 2:
            self.dim = 6 * embed_size + history_dim
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_history, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            path_embed = torch.tensor(np.concatenate([user_embed, node_embed]).astype(np.float32))
            return torch.cat((user_history, path_embed))
        elif self.history_len == 1:
            path_embed = torch.tensor(np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed]).astype(np.float32))
            return torch.cat((user_history, path_embed))
        elif self.history_len == 2:
            path_embed = torch.tensor(np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed]).astype(np.float32))
            return torch.cat((user_history, path_embed))
        else:
            raise Exception('mode should be one of {full, current}')
    

class BatchKGEnvironment(object):
    def __init__(self, data_path, kg_args, max_acts, max_path_len=3, state_history=1, history_dim=32, reward_function="cosine", use_pattern=False, use_enroll=True):
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.kg = load_kg(data_path)
        self.embeds = load_embed(data_path)
        self.embed_size = self.embeds['user'].shape[1]
        self.embeds[kg_args.self_loop] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=state_history, history_dim=history_dim)
        self.state_dim = self.state_gen.dim
        self.train_labels = load_labels(data_path, 'train') 
        self.train_target_labels = load_labels(data_path, 'train_target')
        self.use_pattern = use_pattern
        self.kg_args = kg_args

        self.max_n_bleu = 2
        self.bleu_weights = [1/self.max_n_bleu]*self.max_n_bleu

        ####
        self.reward_fn = reward_function
        self._reward_fns = {
            # 'cosine': self.cosine_reward,
            # 'cosine_train': self.cosine_train_reward,
            # 'cosine_on_path': self.cosine_on_path,
            # 'binary_course': self.binary_course_reward,
            'binary_train': self.binary_train_reward,
            # 'cosine_binary': self.cosine_on_path_binary_final,

            'sequential_binary_cosine': self.sequential_binary_cosine,
            'sequential_cosine': self.sequential_cosine,
            'sequential_bleu': self.sequential_bleu,
            'sequential_binary': self.sequential_binary,
            'sequential_combination': self.seq_combination_reward
        }
        ####

        # Compute user-product scores for scaling.
        u_p_scores = np.dot(self.embeds['user'] + self.embeds[self.kg_args.interaction][0], self.embeds['item'].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

        # Compute path patterns
        self.patterns = []
        for pattern_id in kg_args.path_pattern.keys():
            pattern = kg_args.path_pattern[pattern_id]
            pattern = [self.kg_args.self_loop] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            self.patterns.append(tuple(pattern))

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False
        # For sequential recommendation
        self.hist_state = None
        self.predictions = None
        self.curr_uids = None
        self.use_enroll = use_enroll

    def sequential_binary_cosine(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        uid = path[0][-1]

        if len(path) < 3:
            return 0.0
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        _, curr_node_type, curr_node_id = path[-1]

        if curr_node_type != 'item':
            return 0.0
        
        # if curr_node_id not in self.train_target_labels[uid]:
        #     return 0.0

        self_loops = 0
        for node in path[1:]:
            relation, _, _ = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0

        if curr_node_id in self.train_target_labels[uid]:
            return 1.0

        target_score = 0
        u_vec = self.embeds['user'][uid] + self.embeds[self.kg_args.interaction][0]
        p_vec = self.embeds['item'][curr_node_id]
        score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
        target_score = max(score, 0.0)

        return target_score
    
    def sequential_binary(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        uid = path[0][-1]

        if len(path) < 3:
            return 0.0
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        _, curr_node_type, curr_node_id = path[-1]

        if curr_node_type != 'item':
            return 0.0
        
        if curr_node_id not in self.train_target_labels[uid]:
            return 0.0

        self_loops = 0
        for node in path[1:]:
            relation, _, _ = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0
        return 1.0
    
    def sequential_cosine(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        uid = path[0][-1]

        if len(path) < 3:
            return 0.0
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        _, curr_node_type, curr_node_id = path[-1]

        target_score = 0
        if curr_node_type == 'item':
            if curr_node_id in self.train_labels:
                # Give soft reward for other reached products.
                u_vec = self.embeds['user'][uid] + self.embeds[self.kg_args.interaction][0]
                p_vec = self.embeds['item'][curr_node_id]
                score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
                target_score = max(score, 0.0)

        self_loops = 0
        for node in path[1:]:
            relation, _, _ = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0

        return target_score

    def sequential_bleu(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        learner_id = path[0][-1]

        if len(path) < self.max_num_nodes:
            return 0.0
        
        #print(path)
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0
        
        ref = self.train_target_labels[learner_id]
        ref = self.create_string_targets(ref)

        path = list(filter(lambda p: True if p[0] != self.kg_args.self_loop else False, path))

        pred = []

        for node in path:
            _, type, eid = node
            if(type == 'item'):
                pred.append(str(eid))

        pred_items = pred[1:]  # remove first item user interacted with
        score_all = 0
        num_ngrams = 2
        for i in range(num_ngrams):
            weights = [0] * (i+1)
            weights[i] = 1
            score = bleu_score([pred_items], [[ref]], max_n=i+1, weights=weights) 
            score_all += score
            # if score != 0.0:
            #     print(f'add {bleu_score([pred_items], [[ref]], max_n=i+1, weights=weights), pred_items, ref } to score')
        score_all /= num_ngrams
        return score_all
    
    def seq_combination_reward(self, path):
        cos_score = self.sequential_cosine(path)
        bleu_score = self.sequential_bleu(path)

        return cos_score + bleu_score
    
    def create_string_targets(self, ref):
        return list(map(lambda x: str(x), ref))

    def cosine_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) < 3:
            return 0.0

        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == 'item':
            # Give soft reward for other reached products.
            uid = path[0][-1]
            u_vec = self.embeds['user'][uid] + self.embeds[self.kg_args.interaction][0]
            p_vec = self.embeds['item'][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            target_score = max(score, 0.0)
        return target_score

    def cosine_train_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        if len(path) < 3:
            return 0.0

        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        target_score = 0.0
        learner_id = path[0][-1]
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == 'item':
            if curr_node_id not in self.train_labels[learner_id]:
                return 0.0
            # Give soft reward for other reached products.
            uid = path[0][-1]
            u_vec = self.embeds['user'][uid] + self.embeds[self.kg_args.interaction][0]
            p_vec = self.embeds['item'][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            target_score = max(score, 0.0)
        return target_score    
    
    def cosine_on_path(self, path):
        if len(path) < 3:
            return 0.0
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        _, curr_node_type, curr_node_id = path[-1]
        learner_id = path[0][-1]
        target_score = 0.0
        
        if curr_node_type != 'item':
            return 0.0
        
        if curr_node_id not in self.train_labels[learner_id]:
            return 0.0

        self_loops = 0

        for i, node in enumerate(path[1:]):
            _, previous_node_type, previous_node_id = path[i]
            relation, node_type, node_id = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
                continue
            head_relation_vec = self.embeds[previous_node_type][previous_node_id] + self.embeds[relation][0]
            tail_vec = self.embeds[node_type][node_id]
            target_score += np.dot(head_relation_vec, tail_vec) / (np.linalg.norm(head_relation_vec) * np.linalg.norm(tail_vec))
        

        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0
        
        u_vec = self.embeds['user'][learner_id] + self.embeds[self.kg_args.interaction][0]
        p_vec = self.embeds['item'][curr_node_id]
        target_score += np.dot(u_vec, p_vec) / self.u_p_scales[learner_id]
        target_score = max(target_score, 0.0)
        return target_score
    
    def cosine_on_path_binary_final(self, path):
        if len(path) < 3:
            return 0.0
        
        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        _, curr_node_type, curr_node_id = path[-1]
        learner_id = path[0][-1]
        target_score = 0.0
        
        if curr_node_type != 'item':
            return 0.0
        
        if curr_node_id not in self.train_labels[learner_id]:
            return 0.0

        self_loops = 0
        
        target_score = 0.0
        for i, node in enumerate(path[1:]):
            _, previous_node_type, previous_node_id = path[i]
            relation, node_type, node_id = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
                continue
            head_relation_vec = self.embeds[previous_node_type][previous_node_id] + self.embeds[relation][0]
            tail_vec = self.embeds[node_type][node_id]
            target_score += np.dot(head_relation_vec, tail_vec) / (np.linalg.norm(head_relation_vec) * np.linalg.norm(tail_vec))
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0
        
        if curr_node_id in self.train_labels[learner_id]:
            target_score += 3.0

        target_score = max(target_score, 0.0)
        return target_score
    
    def binary_course_reward(self, path):
            # If it is initial state or 1-hop search, reward is 0.
        if len(path) < 3:
            return 0.0

        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0
            
        self_loops = 0
        for node in path[1:]:
            relation, _, _ = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
                continue
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0

        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]
        if curr_node_type == 'item':
            target_score = 1.0

        return target_score
    
    def binary_train_reward(self, path):
        # If it is initial state or 1-hop search, reward is 0.
        learner_id = path[0][-1]

        if len(path) < 3:
            return 0.0

        if self.use_pattern:
            if not self._has_pattern(path):
                return 0.0

        target_score = 0.0
        _, curr_node_type, curr_node_id = path[-1]

        if curr_node_type != 'item':
            return 0.0
        
        #if curr_node_id not in self.train_labels[learner_id]:
        #    return 0.0

        self_loops = 0
        for node in path[1:]:
            relation, _, _ = node
            if(relation == self.kg_args.self_loop):
                self_loops += 1
                continue
        
        if len(set(path)) <= 3 and self_loops > 0:
            return 0.0
        if curr_node_type == 'item':
            if curr_node_id in self.train_labels.get(learner_id, []):
                target_score = 1.0

        return target_score


    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]
    
    def _get_actions(self, path, done):
        """Compute actions for current node."""
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(self.kg_args.self_loop, curr_node_id)]  # self-loop must be included.

        # (1) If game is finished, only return self-loop action.
        if done:
            return actions

        # (2) Get all possible edges from original knowledge graph.
        # [CAVEAT] Must remove visited nodes!
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_type = self.kg_args.kg_relation[curr_node_type][r]
            next_node_ids = relations_nodes[r]
            if(next_node_type in ['item', 'user']):
                next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
            if not self.use_enroll:
                if(next_node_type == 'user'):
                    next_node_ids = []
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        # (3) If candidate action set is empty, only return self-loop action.
        if len(candidate_acts) == 0:
            return actions

        # (4) If number of available actions is smaller than max_acts, return action sets.
        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        # (5) If there are too many actions, do some deterministic trimming here!
        user_embed = self.embeds['user'][path[0][-1]]
        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = self.kg_args.kg_relation[curr_node_type][r]
            if next_node_type == 'user':
                src_embed = user_embed
            elif next_node_type == 'item':
                src_embed = user_embed + self.embeds[self.kg_args.interaction][0]
            else:  # BRAND, CATEGORY, RELATED_PRODUCT
                src_embed = user_embed + self.embeds[self.kg_args.interaction][0] + self.embeds[r][0]
            score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
            # This trimming may filter out target products!
            # Manually set the score of target products a very large number.
            # if next_node_type == PRODUCT and next_node_id in self._target_pids:
            #    score = 99999.0
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_state(self, i, path, predictions):
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation]."""
        user_embed = self.embeds['user'][path[0][-1]]
        user_id = path[0][-1]
        zero_embed = np.zeros(self.embed_size)
        user_history = self.hist_state[user_id]
        
        # uncomment when training with num_iter > 1
        # last_pred = np.zeros(self.embed_size)
        # if len(predictions) > 0:
        #     last_pred = self.embeds['item'][predictions[-1]]
        if len(path) == 1:  # initial state
            # state = self.state_gen(user_history, user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed, last_pred)
            state = self.state_gen(user_history, user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]  # this can be self-loop!
        if len(path) == 2:
            state = self.state_gen(user_history, user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed,
                                   zero_embed)
            return state

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        state = self.state_gen(user_history, user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    def _batch_get_state(self, batch_path, predictions=None):
        if predictions:
            batch_state = [self._get_state(i, path, predictions[i]) for i, path in enumerate(batch_path)]
        else:
            batch_state = [self._get_state(i, path, []) for i, path in enumerate(batch_path)]
        return torch.stack(batch_state, dim=0)  # [bs, dim]

    def _get_reward(self, path):
        if(self.reward_fn not in self._reward_fns):
            raise "Wrong reward type."
        return self._reward_fns[self.reward_fn](path)   

    def _batch_get_reward(self, batch_path):
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset_predictions(self, size):
        self.predictions = [[] for i in range(size)]

    def reset(self, uids=None, hist_state = None):
        if uids is None:
            all_uids = list(self.kg('user').keys())
            uids = [random.choice(all_uids)]
        self.curr_uids = uids

        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[(self.kg_args.self_loop, 'user', uid)] for uid in uids]
        self._done = False
        self.hist_state = {}
        for i, uid in enumerate(uids):
            self.hist_state[uid] = hist_state[i]
        self._batch_curr_state = self._batch_get_state(self._batch_path, self.predictions)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            if relation == self.kg_args.self_loop:
                next_node_type = curr_node_type
            else:
                next_node_type = self.kg_args.kg_relation[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path, self.predictions)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)
    
    def batch_action_embeddings(self, path_pool=None, act_pool=None):
        """Return action masks of size [bs, 2*embed_size, act_dim]."""
        if path_pool is None:
            path_pool = self._batch_path
        if act_pool is None:
            act_pool = self._batch_curr_actions

        batch_embeddings = []
        
        for actions, path in zip(act_pool, path_pool):
            _, curr_node_type, curr_node_id = path[-1]
            action_embeds = np.zeros((self.act_dim, 2*self.embed_size), dtype=np.uint8)
            relation_embed = self.embeds[self.kg_args.self_loop][0]
            node_embed = self.embeds[curr_node_type][curr_node_id]
            action_embeds[0] = np.concatenate((relation_embed, node_embed))
            for i, action in enumerate(actions[1:]):
                relation = action[0]
                node_id = action[1]
                relation_embed = self.embeds[relation][0]
                node_type = self.kg_args.kg_relation[curr_node_type][relation]
                node_embed = self.embeds[node_type][node_id]
                action_embeds[i + 1] = np.concatenate((relation_embed, node_embed))
            batch_embeddings.append(np.transpose(action_embeds))
        
        return np.stack(batch_embeddings)  
    
    def extract_predictions(self):
        for i, path in enumerate(self._batch_path):
            self_loops = 0
            for node in path[1:]:
                relation, _, _ = node
                if(relation == self.kg_args.self_loop):
                    self_loops += 1

            if len(set(path)) <= 3 and self_loops > 0:
                continue

            _, node_type, node_id = path[-1]
            if(node_type == 'item'):
                self.predictions[i].append(node_id)
        #print(f'Extracting... {self.predictions}')

    def get_predictions(self):
        return self.predictions

    def get_batch_reward(self):
        preds = [self.create_string_targets(p) for p in self.predictions]
        targets = [[] for i in range(len(self.curr_uids))]
    
        for i, uid in enumerate(self.curr_uids):
            targets[i] = self.create_string_targets(self.train_target_labels[uid])

        #print(list(zip(self.curr_uids, preds, targets)))
        score = 0
        num_ngrams = 2

        rewards = []
        for p, t in zip(preds, targets):
            #print(p,t)
            score = 0.0
            # BLEU score
            for i in range(num_ngrams):
                weights = [0] * (i+1)
                weights[i] = 1
                score += bleu_score([p], [[t]], max_n=self.max_n_bleu, weights=self.bleu_weights) 
            score /= num_ngrams
            
            #for i in range(min(len(p), len(t))):
            #    p_vec = self.embeds['item'][int(p[i])]
            #    t_vec = self.embeds['item'][int(t[i])]
            #    score += np.dot(t_vec, p_vec) / (np.linalg.norm(t_vec) * np.linalg.norm(p_vec))
            rewards.append(score)
        #print(rewards)
        return rewards

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)
