# Sequential course recommendation

This is the repository for sequential course recommender system developed as part of the Research project in ML4ED lab.


## Run the code

### Setup

1. Clone this repository to your local machine.
2. Navigate to the recommender directory.

### Data preparation

1. Obtain data from ic1files and place it in the `recommender\data\mooc\files` directory.
2. Create a `config.json` file with the specified parameters using the template provided in the `configs` directory. You can generate config files for various scenarios using the `create_configs.py` script. The script produces config files with specified parameters for the next-item task based on template config file. Example for generating configs for next-batch scenario is given in the comment in the script.


### Training and testing
Run the following scripts with provided config file as the command line parameter:

```
python preprocess.py --config "path_to_config_file"
```

```
python train_transe_model.py --config "path_to_config_file"
```
```
python train_agent.py --config "path_to_config_file"
```
```
python test_agent.py --config "path_to_config_file"
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
docker push ic-registry.epfl.ch/d-vet/your_name/"image_name":"tag"
```
4.  Login to RunAI and launch the job:
```
runai submit --name experiment_name -i ic-registry.epfl.ch/d-vet/your_name/image_name:tag -e "CONFIG=--config ./configs/config_name.json " '
```

**NOTE**: For optimizing the resources on the cluster, Docker image runs only training and testing scripts for the RL agent. Both scripts are run on CPU. Preprocessing and training the KG embeddings can be run only once prior to launching jobs on the cluster.
In order to run full pipeline in the container, uncomment execution commands in the `entrypoint.sh` script.

## Baselines

Follow the steps in the README.md file provided in the `sequential_baselines` directory.

