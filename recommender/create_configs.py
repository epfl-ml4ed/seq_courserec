import json

def set_all(data, key, value):
    data['PREPROCESS'][key] = value
    data['TRAIN_EMBEDS'][key] = value
    data['TRAIN_AGENT'][key] = value
    data['TEST_AGENT'][key] = value

def set_agent(data, key, value):
    data['TRAIN_AGENT'][key] = value
    data['TEST_AGENT'][key] = value 

for dataset in ['mooc']:

    with open(f'./configs/config_{dataset}_template.json', 'r') as f:
        data = json.load(f)

    set_all(data, 'use_wandb', True)

    for use_enroll in [('with_enroll', True)]: #, ('no_enroll', False)
        exp_dir = 1
        for seed in [23, 24, 25]:
            for epochs in [100]:
                for max_path_len, topk, next_item, tmp_dir, project_name in [ (3, [25, 5, 1], True, f'./{dataset}/next_item', 'item')]:#, (7, [1,1,1,1,1,1,1], False, f'./{dataset}_results/next_batch', 'batch')]:
                    for state_history in [0]:
                        for reward in ['sequential_bleu', 'sequential_cosine', 'sequential_binary']: #, 'sequential_combination']:
                            set_agent(data, 'seed', seed)
                            set_agent(data, 'epochs', epochs)
                            set_agent(data, 'max_path_len', max_path_len)
                            set_agent(data, 'state_history', state_history)
                            set_agent(data, 'reward', reward)
                            set_agent(data, 'exp_dir', str(exp_dir))
                            set_agent(data, 'use_enroll', use_enroll[1])
                            set_all(data, 'tmp_dir', tmp_dir)
                            set_all(data, 'wandb_project_name', f'{dataset}-{project_name}')

                            data['TEST_AGENT']['topk'] = topk
                            data['TEST_AGENT']['next_item'] = next_item

                            if(max_path_len == 3):
                                data['PREPROCESS']['seq_batch_size'] = 1

                            i = 1
                            for args in ['PREPROCESS', 'TRAIN_EMBEDS', 'TRAIN_AGENT', 'TEST_AGENT']:
                                data[args]['wandb_run_name'] = f'{exp_dir}_{i}'
                                i += 1

                            with open(f'./configs/mooc_{use_enroll}/{exp_dir}.json', 'w') as f:
                                json.dump(data, f)
                                
                            exp_dir += 1