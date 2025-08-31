from omegaconf import DictConfig

from transformers import AutoTokenizer, PreTrainedTokenizer


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_data_encoder(self) -> PreTrainedTokenizer:
        data_encoder = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name,
            use_fast=True,
            revision=self.config.revision,
        )

        if data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                self.config.reference_data_encoder_name
            )
            data_encoder.chat_template = reference_data_encoder.chat_template

        if data_encoder.pad_token_id is None:
            data_encoder.pad_token_id = data_encoder.eos_token_id
        if self.config.left_padding:
            data_encoder.padding_side = "left"
        else:
            data_encoder.padding_side = "right"

        return data_encoder
