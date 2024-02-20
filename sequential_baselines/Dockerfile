FROM continuumio/anaconda3

RUN apt update

ENV WANDB_ENTITY=
ENV WANDB_API_KEY=

COPY ./environment_baselines.yml /

# Create the environment:
RUN conda env create -f environment_baselines.yml

COPY . /
COPY ./entrypoint.sh /

RUN chmod u+x entrypoint.sh

# Make RUN commands use the new environment:
RUN echo "conda activate baselines" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV ARGS=""

ENTRYPOINT ["./entrypoint.sh"]
