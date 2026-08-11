import torch
import torch.nn as nn
from DiTAR.model.backbones.qwen3 import Qwen3ModelForMultiModal, Qwen3ForCausalLM, Qwen3Config

class CausalAR_Qwen(nn.Module):
    def __init__(
        self,
        version,
        qwen_config_path,
        pretrained_LM_path,
        load_pretrained_weights,
    ):
        super().__init__()

        if version == "qwen3":
            config = Qwen3Config.from_pretrained(qwen_config_path)
            self.model = Qwen3ModelForMultiModal(config)
            if load_pretrained_weights:
                print(f"Loading pre-trained weights from '{pretrained_LM_path}'...")
                pretrained_model = Qwen3ForCausalLM.from_pretrained(pretrained_LM_path)
                loading_info=self.model.load_state_dict(pretrained_model.model.state_dict())
                print(loading_info)
                print("✅ Pre-trained weights loaded successfully into custom model.")
        else:
            raise ValueError(f"Invalid version: {version}")

    def forward(
        self,
        inputs_embed: torch.Tensor,
        modality_type_ids: torch.Tensor,
        padding_mask=None,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds = inputs_embed,
            modality_type_ids = modality_type_ids,
            attention_mask = padding_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

    def inference(
        self,
        inputs_embed: torch.Tensor,
        modality_type_ids: torch.Tensor,
        padding_mask = None,
        past_key_values = None,
        use_cache = None,
    ) -> torch.Tensor:

        outputs = self.model(
            inputs_embeds = inputs_embed,
            modality_type_ids = modality_type_ids,
            attention_mask = padding_mask,
            output_hidden_states=True,
            past_key_values = past_key_values,
            use_cache = use_cache,
        )

        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        return last_hidden_state, past_key_values
