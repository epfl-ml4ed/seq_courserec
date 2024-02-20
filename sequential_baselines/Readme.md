# Sequential baselines

## Run the baselines

### Data preparation

1. Obtain data from ic1files and place it in the `datasets` directory.
2. Create a `config.json` files with the specified parameters. Configs for next-item task are given as an example in the `configs` directory.


### Training and testing
Here is the example for running the baselines on mooc data for next-item task.


1. Login to Weights and Biases
2. Format the data and create atomic files used in Recbole library:
```
python format_mooc.py --datadir ./datasets/mooc_item/files
```
3. Run the main script with config file provided as the argument. The results for the baseline method computed on multiple seeds will be saved as artifact (table) in the W&B project. 
```
python main.py --config ./configs/run_all_item_gru.yaml --project_name "wandb_project_name" --run_name "wandb_run_name"
```


## Run code in Docker container on RunAI cluster:
If using Weights and Biases, include your name and API key in the Dockerfile.

1. Build the Docker image:
```
docker build -t "image_name":"tag" .
```
2. Tag the image to ic-registry:
```
docker tag "image_name":"tag" ic-registry.epfl.ch/d-vet/"your_name"/"image_name":"tag"
```
3. Push the image to the registry: 
```
docker push ic-registry.epfl.ch/d-vet/"your_name"/"image_name":"tag"
```
4.  Login to RunAI and launch the job:
```
runai submit --name experiment_name -i ic-registry.epfl.ch/d-vet/your_name/image_name:tag -e "ARGS=--config ./configs/run_all_item_gru.yaml --project_name wandb_project_name --run_name wandb_run_name" -g 0.05'
```

## Installation
 RecBole