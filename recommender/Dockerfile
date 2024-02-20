FROM continuumio/anaconda3

RUN apt update

ENV WANDB_ENTITY=
ENV WANDB_API_KEY=

COPY ./environment_pgpr.yml /

# Create the environment:
RUN conda env create -f environment_pgpr.yml

COPY . /
COPY ./entrypoint.sh /

RUN chmod u+x entrypoint.sh

# Make RUN commands use the new environment:
RUN echo "conda activate pgpr" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV CONFIG=""

ENTRYPOINT ["./entrypoint.sh"]
