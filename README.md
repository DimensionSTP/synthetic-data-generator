# Synthetic data generating pipeline

## For synthetic data generation

### Dataset

Any Structured Datasets

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/synthetic-data-generator.git
cd synthetic-data-generator

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Generate

* end-to-end

```shell
python main.py mode=generate
```

### Examples of shell scipts

* generate

```shell
bash scripts/generate.sh
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__
