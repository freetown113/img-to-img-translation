 

# img-to-img-translation project

## To launch the project:
It's strictly advisable to create a container from docker file or with docker compose from the Docker directory.

In environment do the follow:
```
cd /
git clone https://github.com/freetown113/img-to-img-translation.git
cd img-to-img-translation
# Make changes to config/parameters.yaml if necessary

# Launch project
python3 main
```

Dataset will be downloaded and unpacked automatically. Immediately after it the learning process will start. 


## Project details: 
Realisation of the SimpleViT used in blocks/vit.py is taken from  https://github.com/lucidrains/vit-pytorch

Learning process with current parameters was tested on a single Nvidia RTX2080 Ti. To make it work on the devices 
with smaller amount of memory one should reduce batch size and/or make "test_while_training: False" in config/parameters.yaml 



