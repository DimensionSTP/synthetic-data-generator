import os

from omegaconf import DictConfig

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

    file_name = f"{config.dataset_name}.csv"
    full_data_path = os.path.join(
        config.data_dir,
        file_name,
    )

    try:
        df = pd.read_csv(
            full_data_path,
            encoding="cp949",
        )
        df.columns = [config.document_column_name, config.category_column_name]
    except Exception as e:
        wandb.run.alert(
            title="Data Loading Error",
            text=f"Failed to load {full_data_path} with cp949 encoding: {e}",
            level="ERROR",
        )
        raise e

    df = df.fillna("")

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

    results = []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA pairs"):
            category = row[config.category_column_name]
            document = row[config.document_column_name]

            if not document or not category:
                continue

            question_gen_prompt = config.prompt.question.format(
                category=category,
                document=document,
            )

            conversation_q = [
                {
                    "role": "user",
                    "content": question_gen_prompt,
                }
            ]
            prompt_q = data_encoder.apply_chat_template(
                conversation=conversation_q,
                tokenize=False,
                add_generation_prompt=True,
            )

            outputs_q = llm.generate(
                prompts=[prompt_q],
                sampling_params=sampling_params,
            )
            generated_questions_text = outputs_q[0].outputs[0].text.strip()
            questions = [
                q.strip() for q in generated_questions_text.split("\n") if q.strip()
            ]

            for question in questions[: config.gen_num_questions]:
                answer_gen_prompt = config.prompt.answer.format(
                    document=document,
                    question=question,
                )

                conversation_a = [
                    {
                        "role": "user",
                        "content": answer_gen_prompt,
                    }
                ]
                prompt_a = data_encoder.apply_chat_template(
                    conversation=conversation_a,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                outputs_a = llm.generate(
                    prompts=[prompt_a],
                    sampling_params=sampling_params,
                )
                answer = outputs_a[0].outputs[0].text.strip()

                results.append(
                    {
                        config.context_column_name: document,
                        config.question_column_name: question,
                        config.answer_column_name: answer,
                    }
                )

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

        wandb.log({"synthetic_qa_results": wandb.Table(dataframe=result_df)})
        wandb.run.alert(
            title="Synthetic QA Generation Complete",
            text=f"QA generation on {config.dataset_name} has successfully finished.",
            level="INFO",
        )

    except Exception as e:
        wandb.run.alert(
            title="Synthetic QA Generation Error",
            text=f"An error occurred during generation on {config.dataset_name}: {e}",
            level="ERROR",
        )
        if results:
            os.makedirs(config.output_dir, exist_ok=True)
            output_path = os.path.join(
                config.output_dir,
                f"partial_{config.output_name}.jsonl",
            )
            pd.DataFrame(results).to_json(
                output_path,
                orient="records",
                lines=True,
                force_ascii=False,
            )
        raise e
