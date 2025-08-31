import os

from omegaconf import DictConfig

import json

import pandas as pd

import torch

from transformers import set_seed

import wandb

from tqdm import tqdm

from vllm import LLM, SamplingParams

from huggingface_hub import snapshot_download

from ..utils import SetUp


def generate_synthetic_qa(
    config: DictConfig,
) -> None:
    wandb.init(
        project=config.project_name,
        name=config.model_type,
    )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()

    num_gpus = torch.cuda.device_count()

    try:
        llm = LLM(
            model=config.pretrained_model_name,
            tokenizer=config.pretrained_model_name,
            revision=config.revision,
            tensor_parallel_size=num_gpus,
            seed=config.seed,
            trust_remote_code=True,
        )
    except Exception:
        model_path = snapshot_download(
            repo_id=config.pretrained_model_name,
            revision=config.revision,
        )
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=num_gpus,
            seed=config.seed,
            trust_remote_code=True,
        )

    if config.do_sample:
        generation_config = config.generation_config
    else:
        generation_config = {
            "temperature": 0,
            "top_p": 1,
        }

    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        skip_special_tokens=True,
        stop_token_ids=[data_encoder.eos_token_id],
        **generation_config,
    )

    file_name = f"{config.dataset_name}.{config.dataset_format}"
    full_data_path = os.path.join(
        config.data_dir,
        file_name,
    )

    if config.dataset_format == "csv":
        df = pd.read_csv(full_data_path)
    elif config.dataset_format == "parquet":
        df = pd.read_parquet(full_data_path)
    elif config.dataset_format in ["json", "jsonl"]:
        df = pd.read_json(
            full_data_path,
            lines=True if config.dataset_format == "jsonl" else False,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {config.dataset_format}")

    df = df.fillna("_")

    prompts = []
    labels = []

    for _, row in df.iterrows():
        instruction = row[config.instruction_column_name].strip()
        data = row[config.data_column_name].strip()
        label = row[config.target_column_name].strip()

        conversation = [
            {
                config.role_column_name: "system",
                config.content_column_name: instruction,
            },
            {
                config.role_column_name: "user",
                config.content_column_name: data,
            },
        ]

        prompt = data_encoder.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(label)

    try:
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

        results = []
        for output, label in zip(outputs, labels):
            instruction = output.prompt
            generation = output.outputs[0].text.strip()
            results.append(
                {
                    "instruction": instruction,
                    "generation": generation,
                    "label": label,
                }
            )

        os.makedirs(
            config.output_dir,
            exist_ok=True,
        )
        test_output_path = os.path.join(
            config.output_dir,
            f"{config.output_name}.json",
        )

        df = pd.DataFrame(results)
        df.to_json(
            test_output_path,
            orient="records",
            indent=2,
            force_ascii=False,
        )

        wandb.log({"test_results": wandb.Table(dataframe=df)})

        wandb.run.alert(
            title="vLLM Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        wandb.run.alert(
            title="vLLM Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e


def test_vllm_multi_turn(
    config: DictConfig,
) -> None:
    wandb.init(
        project=config.project_name,
        name=config.model_type,
    )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()

    num_gpus = torch.cuda.device_count()

    try:
        llm = LLM(
            model=config.pretrained_model_name,
            tokenizer=config.pretrained_model_name,
            revision=config.revision,
            tensor_parallel_size=num_gpus,
            seed=config.seed,
            trust_remote_code=True,
        )
    except Exception:
        model_path = snapshot_download(
            repo_id=config.pretrained_model_name,
            revision=config.revision,
        )
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=num_gpus,
            seed=config.seed,
            trust_remote_code=True,
        )

    model_max_len = llm.llm_engine.model_config.max_model_len

    if config.do_sample:
        generation_config = config.generation_config
    else:
        generation_config = {
            "temperature": 0,
            "top_p": 1,
        }

    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        skip_special_tokens=True,
        stop_token_ids=[data_encoder.eos_token_id],
        **generation_config,
    )

    file_name = f"{config.dataset_name}.{config.dataset_format}"
    full_data_path = os.path.join(
        config.data_dir,
        file_name,
    )

    if config.dataset_format == "csv":
        df = pd.read_csv(full_data_path)
    elif config.dataset_format == "parquet":
        df = pd.read_parquet(full_data_path)
    elif config.dataset_format in ["json", "jsonl"]:
        df = pd.read_json(
            full_data_path,
            lines=True if config.dataset_format == "jsonl" else False,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {config.dataset_format}")

    df = df.fillna("_")

    try:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
            contents = row[config.content_column_name]

            if isinstance(contents, list):
                conversation = []
                generations = []

                for content in contents:
                    conversation.append(
                        {
                            config.role_column_name: "user",
                            config.content_column_name: content,
                        }
                    )
                    prompt = data_encoder.apply_chat_template(
                        conversation=conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    prompt_token_ids = data_encoder.encode(prompt)
                    if len(prompt_token_ids) >= model_max_len:
                        print(
                            f"Prompt length ({len(prompt_token_ids)}) is exceeding model max length ({model_max_len}). "
                            f"Skipping this turn."
                        )
                        generation = "MODEL_MAX_LENGTH_EXCEEDED"
                        generations.append(generation)
                        break

                    output = llm.generate(
                        prompts=[prompt],
                        sampling_params=sampling_params,
                    )[0]
                    generation = output.outputs[0].text.strip()
                    generations.append(generation)

                    conversation.append(
                        {
                            config.role_column_name: "assistant",
                            config.content_column_name: generation,
                        }
                    )

                result_item = row.to_dict()
                result_item["generation"] = generations
                results.append(result_item)
            else:
                conversation = [
                    {
                        config.role_column_name: "user",
                        config.content_column_name: contents,
                    }
                ]
                prompt = data_encoder.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                output = llm.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                )[0]
                generation = output.outputs[0].text.strip()

                result_item = row.to_dict()
                result_item["generation"] = generation
                results.append(result_item)

        os.makedirs(
            config.output_dir,
            exist_ok=True,
        )
        output_path = os.path.join(
            config.output_dir,
            f"{config.output_name}.jsonl",
        )

        result_df = pd.DataFrame(results)
        result_df.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        for column in result_df.columns:
            result_df[column] = result_df[column].apply(
                lambda value: (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                    if isinstance(value, (list, dict, set))
                    else value
                )
            )

        wandb.log({"test_results": wandb.Table(dataframe=result_df)})
        wandb.run.alert(
            title="vLLM Multi-Turn Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )

    except Exception as e:
        wandb.run.alert(
            title="vLLM Multi-Turn Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e
