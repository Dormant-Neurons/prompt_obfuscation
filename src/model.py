import logging

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)
logging.getLogger('bitsandbytes').setLevel(logging.WARNING)

class Model():
    """A wrapper class for loading and interacting with a Hugging Face CausalLM model and its tokenizer."""
    def __init__(
        self,
        name_or_path: str,
        quantization_mode: str | None
    ):
        self.logger = logging.getLogger(__name__)
        self.name_or_path = name_or_path
        self.quantization_mode = quantization_mode
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.word_embedding_layer = None

        self.logger.debug(f"Initializing Model class for: {self.name_or_path}")
        self._load_model_and_tokenizer()

    def _determine_compute_dtype(self) -> torch.dtype:
        """Determines the appropriate compute dtype based on CUDA capability."""
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                self.logger.debug("CUDA capability >= 8.0, using bfloat16 for compute.")
                return torch.bfloat16
        self.logger.debug("CUDA capability < 8.0 or CUDA not available, using float16 for compute.")
        return torch.float16


    def _get_bnb_config(self, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
        """Creates BitsAndBytesConfig if quantization is requested."""
        if self.quantization_mode == "4bit":
            self.logger.debug("Configuring for 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif self.quantization_mode == "8bit":
            self.logger.debug("Configuring for 8-bit quantization.")
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        self.logger.debug("No quantization requested or unsupported mode.")
        return None

    def _load_model_and_tokenizer(self):
        """
        Loads the specified model and its corresponding tokenizer.

        This method handles model loading with optional quantization (4-bit or 8-bit),
        sets the appropriate compute dtype, and loads the tokenizer, including a
        fallback mechanism. It also adds a new pad token and resizes the model's
        token embeddings accordingly.
        """
        self.logger.debug(f"Loading model: {self.name_or_path} with quantization: {self.quantization_mode or 'None'}")

        compute_dtype = self._determine_compute_dtype()
        bnb_config = self._get_bnb_config(compute_dtype)

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = compute_dtype
        

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name_or_path,
                **model_kwargs
            )
            self.logger.debug(f"Model '{self.name_or_path}' loaded successfully.")
            self.model.eval()
            self.device = self.model.device
            self.dtype = self.model.dtype
            self.logger.debug(f"Model loaded on device: {self.device}, with dtype: {self.dtype}")
        except Exception as e:
            self.logger.exception(f"Failed to load model '{self.name_or_path}'. Error: {e}")
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.name_or_path,
                use_fast=False,
                padding_side="left",
                trust_remote_code=True,
            )
            self.logger.debug(f"Tokenizer for '{self.name_or_path}' loaded successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to load tokenizer from '{self.name_or_path}'. Error: {e}")
            try:
                if hasattr(self.model.config, "_name_or_path") and self.model.config._name_or_path and self.model.config._name_or_path != self.model_name_or_path:
                    fallback_path = self.model.config._name_or_path
                    self.logger.warning(f"Attempting to load tokenizer from model's config _name_or_path: {fallback_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        fallback_path,
                        use_fast=False,
                        padding_side="left",
                        trust_remote_code=True,
                    )
                    self.logger.debug(f"Tokenizer successfully loaded from fallback path: {fallback_path}")
                else:
                    raise
            except Exception as fallback_e:
                self.logger.exception(f"Fallback tokenizer loading also failed. Error: {fallback_e}")
                raise 
        
        self.logger.info("Model and tokenizer loaded successfully.")

        new_pad_token = "<|pad|>"
        self.tokenizer.add_special_tokens({"pad_token": new_pad_token})
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.logger.debug("Added pad token and resized token embeddings.")


        self.word_embedding_layer = self.model.get_input_embeddings()

        self.vocab_size = len(self.tokenizer)


    def generate_logits(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        token_count: int,
        embedded: bool,
    ) -> torch.Tensor:
        """
        Generates token logits for a given input.

        This method performs greedy decoding to get the logits for each generated token,
        which are needed for calculating KL-divergence and cross-entropy loss.

        Args:
            input_tensor (torch.Tensor): The input token IDs or embeddings.
            attention_mask (torch.Tensor): The attention mask for the input.
            token_count (int): The number of new tokens for which to generate logits.
            embedded (bool): Whether the input_tensor is embeddings (True) or token IDs (False).

        Returns:
            (torch.Tensor): The output from the model's generate function, including scores.
        """
        model_input_args = {}
        if embedded:
            model_input_args["inputs_embeds"] = input_tensor
        else:
            model_input_args["input_ids"] = input_tensor
        with torch.no_grad():
            output = self.model.generate(
                **model_input_args,
                attention_mask=attention_mask,
                max_new_tokens=token_count, 
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
                return_dict_in_generate=True, 
                output_scores=True,
                do_sample=False,
                num_beams=1,
                top_p = None,
                top_k = None,
                temperature = None,
                return_legacy_cache=False
            )
        return output

    
    def get_embeddings(
        self,
        input_ids: torch.Tensor
    ):
        """Retrieves the embeddings for a given tensor of token IDs."""
        embedding_layer = self.model.get_input_embeddings()
        return embedding_layer.weight[input_ids].cpu()
    
    def get_embedding_matrix(
        self
    ) -> torch.Tensor:
        """Returns the model's entire word embedding matrix."""
        embedding_layer = self.model.get_input_embeddings()
        return embedding_layer.weight


    def generate_output(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_params: dict,
        embedded: bool,
    ) -> torch.Tensor:
        """
        Generates token sequences for a given input using sampling.

        Args:
            input_tensor (torch.Tensor): The input token IDs or embeddings.
            attention_mask (torch.Tensor): The attention mask for the input.
            generation_params (dict): A dictionary of parameters for sampling-based generation.
            embedded (bool): Whether the input_tensor is embeddings (True) or token IDs (False).

        Returns:
            (torch.Tensor): A tensor of generated output sequences.
        """
        with torch.no_grad():
            base_gen_settings = {
                "attention_mask": attention_mask,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "remove_invalid_values": True,
            }
            final_gen_params = {**base_gen_settings, **generation_params}
            model_input_args = {}
            if embedded:
                model_input_args["inputs_embeds"] = input_tensor
            else:
                model_input_args["input_ids"] = input_tensor
            
            output_sequences = self.model.generate(
                **model_input_args,
                **final_gen_params
            ).cpu()

            num_return_sequences = generation_params.get('num_return_sequences', 1)
            current_batch_size = output_sequences.shape[0] // num_return_sequences
            output_sequences = output_sequences.view(current_batch_size, num_return_sequences, -1)

        return output_sequences