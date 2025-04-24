import json
import random
from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from fastchat.model import get_conversation_template, load_model
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from ftfy import fix_text
from transformers.generation import GenerationConfig


from rank_llm.rerank.rankllm import PromptMode, RankLLM
from rank_llm.result import Result


class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of Rank_GPT prompt."
            )
        self._llm = AutoModelForCausalLM.from_pretrained(model, load_in_8bit=True, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(model)

       
        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]
 
    def run_llm( #,
        self, prompt: str, current_window_size: Optional[int] = None, num: Optional[int] = 0
    ) -> Tuple[str, int]:

        if current_window_size is None:
            current_window_size = self._window_size
        
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size, 0)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size, 0)

        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        output_ids = self._llm.generate(**inputs, generation_config=gen_cfg, output_scores=True, return_dict_in_generate=True)

        # dict_keys(['sequences', 'scores', 'past_key_values'])
        list_of_probabilities = []
        to_return = ''
        numbers = [str(number) for number in range(0, 100)] 
        lenght_of_out = len(output_ids['scores'])
        for i in range(len(output_ids['scores'])):
            sco = F.softmax(output_ids['scores'][i], dim=-1)

            t = torch.topk(sco, 1) #prendi solo il massimo
        
            tok = self._tokenizer.decode(t.indices[0][0], skip_special_tokens=True, spaces_between_special_tokens=False)  
            if tok in numbers:
                list_of_probabilities.append(t.values[0])
            to_return+=tok


        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return to_return, lenght_of_out, list_of_probabilities


    def num_output_tokens(self, current_window_size: Optional[int] = None, num: int =0) -> int:

        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            _output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate


    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier [].  Use ONLY the numbers of identifiers present in the passages. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain. You MUST use ONLY the number of identifiers present in the passages, you will be penalized otherwise. I will tip you 300$ if you use only number of identifiers present in the passages\n"

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int, num: int
    ) -> Tuple[str, int]:
        query = result.query
        
        num = len(result.hits[rank_start:rank_end])
        
        max_length = 300 * (20 / (rank_end - rank_start))

        while True:
            conv = get_conversation_template(self._model)
            if self._system_message:
                conv.set_system_message(self._system_message) 
            conv = self._add_few_shot_examples(conv) 
            prefix = self._add_prefix_prompt(query, num)  
            rank = 0
            input_context = f"{prefix}\n"
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                input_context += f"[{rank}] {self._replace_number(content)}\n"

            input_context += self._add_post_prompt(query, num)
            conv.append_message(conv.roles[0], input_context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(
                rank_end - rank_start, num
            ):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens
                        - self.max_tokens()
                        + self.num_output_tokens(rank_end - rank_start, num)
                    )
                    // ((rank_end - rank_start) * 4),
                )
        return prompt, self.get_num_tokens(prompt)
    

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0