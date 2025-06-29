import copy
import random
import torch
import re
import json

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from rank_llm.result import RankingExecInfo, Result

from sentence_transformers import SentenceTransformer, util
class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    LRL = "LRL"

    def __str__(self):
        return self.value


class RankLLM(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        num_few_shot_examples: int,
    ) -> None:
        self._model = model
        self._context_size = context_size
        self._prompt_mode = prompt_mode
        self._num_few_shot_examples = num_few_shot_examples
        self.embedder = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-12-v3")

    def max_tokens(self) -> int:
        """
        Returns the maximum number of tokens for a given model

        Returns:
            int: The maximum token count.
        """
        return self._context_size

    @abstractmethod
    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.
        """
        pass

    @abstractmethod
    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """
        Abstract method to create a prompt based on the result and given ranking range.

        Args:
            result (Result): The result object containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[Union[str, List[Dict[str, str]]], int]: A tuple object containing the generated prompt and the number of tokens in the generated prompt.
        """
        pass


    @abstractmethod
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt for which to compute the token count for.

        Returns:
            int: The number of tokens in the given prompt.
        """
        pass

    @abstractmethod
    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Abstract method to calculate the cost per 1,000 tokens for the target language model.

        Args:
            input_token (bool): Flag to indicate if the cost is for input tokens or output tokens.

        Returns:
            float: The cost per 1,000 tokens.
        """
        pass

    @abstractmethod
    def num_output_tokens(self) -> int:
        """
        Abstract method to estimate the number of tokens in the model's output, constrained by max tokens for the target language model.

        Returns:
            int: The estimated number of output tokens.
        """
        pass
    
    def similarity_query_passages(self, query: str, passage_1: str, passage_2: str, num_1: int, num_2: int) -> int:  #passato
        """
        Given a query and two passages, this function returns the num_passage that is most similar to the query.

        Args:
            query (str): The query string.
            passage_1 (str): The first passage string.
            passage_2 (str): The second passage string.

        Returns:
            str: The passage that is most similar to the query.
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        passage_1_embedding = self.embedder.encode(passage_1, convert_to_tensor=True)
        passage_2_embedding = self.embedder.encode(passage_2, convert_to_tensor=True)
        similarity_1 = util.pytorch_cos_sim(query_embedding, passage_1_embedding)
        similarity_2 = util.pytorch_cos_sim(query_embedding, passage_2_embedding)
        if similarity_1 > similarity_2:
            return num_1
        else:
            return num_2


    def save_lists_to_file(self, lists, filename):
        with open(filename, 'a') as file:
            for i, lst in enumerate(lists):
                json.dump(lst, file)
                if i < len(lists) - 1:  # Check if it's not the last list
                    file.write(', ')
            file.write('\n')


    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> Result:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """

        prompt_0, in_token_count = self.create_prompt(result, rank_start, rank_end, 0)

        
        permutation, out_token_count, list_of_probabilities = self.run_llm(
            prompt_0, current_window_size=rank_end - rank_start)
       
        numbers = [str(numero) for numero in range(0, 34)]
        per = []

        
        if permutation[0] == '>':
            permutation = permutation[1:]
      
        permutation = permutation.replace(' ', '')
        permutation =  permutation.split('>')

       
        for i in range(len(permutation)):
            if len(permutation[i]) < 3:
                continue


            if permutation[i][0] != '[' :
                    permutation[i] = '[' + permutation[i]
            if permutation[i][-1] != ']':
                    permutation[i] = permutation[i] + ']'
            app = permutation[i].split(']')[0].split('[')[1]
            if app in numbers:
                per.append(int(app))




        ######################################### TO SAVE THE OUTPUTS ################################################################
        # self.save_lists_to_file(per, 'YOURPATH.TXT')
        # lis = []
        # for i in list_of_probabilities:
        #     lis.append(i.to('cpu').item())
        # self.save_lists_to_file(lis, 'YOURPATH.TXT')
        ##################################################################################################################

        list_passages = list(dict.fromkeys(per))

        while len(list_passages) < rank_end:
            for i in range(1, rank_end+1):
                if i not in list_passages:
                    list_passages.append(i)
        permutation = " > ".join(f"[{str(value)}]" for value in list_passages)

        if logging:
            print(f"output: {permutation}")
        ranking_exec_info = RankingExecInfo(
            prompt_0, permutation, in_token_count, out_token_count
        )
        if result.ranking_exec_summary == None:
            result.ranking_exec_summary = []
        result.ranking_exec_summary.append(ranking_exec_info)
        result = self.receive_permutation(result, permutation, rank_start, rank_end)
        return result



    def permutation_pipeline_from_file(self, listresults, result: Result, rank_start: int, rank_end: int, logging: bool = False) -> Result:
        prompt, in_token_count = self.create_prompt(result, rank_start, rank_end, 0)

        ranking_exec_info = RankingExecInfo(
            prompt, listresults, in_token_count, len(listresults.replace(' ', ''))
        )
        if result.ranking_exec_summary == None:
            result.ranking_exec_summary = []
        result.ranking_exec_summary.append(ranking_exec_info)
        result = self.receive_permutation(result, listresults, rank_start, rank_end)
        return result
    
    def sliding_windows(
        self,
        retrieved_result: Result,
        rank_start: int,
        rank_end: int,
        window_size: int, #20
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        cnt: int = 0
    ) -> Result:
        """
        Applies the sliding window algorithm to the reranking process.

        Args:
            retrieved_result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after applying the sliding window technique.
        """
        #######################################TO RUN EVALUATION OF AGGREGATION METHODS################################################################################
        with open("results/best_two_models.txt", "r") as file1:  
            file1_lines = file1.readlines()
        file1_numbers = []
        file1_probabilities = []

        for i, line in enumerate(file1_lines):
            file1_numbers.append([int(num) for num in line.strip().split(',')])
        # # for i, line in enumerate(file1_lines):
        # #     if i % 2 == 0:  # Even lines, for numbers
        # #         file1_numbers.append([int(num) for num in line.strip().split(',')])
        # #     else:  # Odd lines, for probabilities
        # #         file1_probabilities.append([float(prob) for prob in line.split(',')])
        file1_numb = []
        for i in file1_numbers:
            permutation = " > ".join(f"[{str(value)}]" for value in i)
            file1_numb.append(permutation)
        #######################################################################################################################

        rerank_result = copy.deepcopy(retrieved_result)
        if shuffle_candidates:
            # First randomly shuffle rerank_result between rank_start and rank_end
            rerank_result.hits[rank_start:rank_end] = random.sample(
                rerank_result.hits[rank_start:rank_end],
                len(rerank_result.hits[rank_start:rank_end]),
            )
            # Next rescore all candidates with 1/rank
            for i, hit in enumerate(rerank_result.hits):
                hit["score"] = 1.0 / (i + 1)
                hit["rank"] = i + 1
        end_pos = rank_end
        start_pos = rank_end - window_size
        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while end_pos > rank_start and start_pos + step != rank_start:
            ############ DECOMMENT IF YOU WANT TO USE THE STANDARD METHODS ################################################
            # start_pos = max(start_pos, rank_start)
            # rerank_result = self.permutation_pipeline(
            #     rerank_result, start_pos, end_pos, logging
            # )
            
            ######### UNCOMMENT IF YOU WANT TO USE THE AGGREGATION METHODS ##################################################
            rerank_result = self.permutation_pipeline_from_file(file1_numb[cnt], rerank_result, start_pos, end_pos, logging)
            end_pos = end_pos - step
            start_pos = start_pos - step
        return rerank_result

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - window_size) / step + 1
        input_token_count = (
            num_q * num_promt * (self._context_size - self.num_output_tokens())
        )
        output_token_count = num_q * num_promt * self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def get_ranking_cost(
        self,
        retrieved_results: List[Dict[str, Any]],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Dict[str, Any]]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            end_pos = rank_end
            start_pos = rank_end - window_size
            while start_pos >= rank_start:
                start_pos = max(start_pos, rank_start)
                prompt, _ = self.create_prompt(result, start_pos, end_pos)
                input_token_count += self.get_num_tokens(prompt)
                end_pos = end_pos - step
                start_pos = start_pos - step
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, result: Result, permutation: str, rank_start: int, rank_end: int
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.hits' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.hits[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response):
            result.hits[j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result.hits[j + rank_start]:
                result.hits[j + rank_start]["score"] = cut_range[j]["score"]
        return result

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)
