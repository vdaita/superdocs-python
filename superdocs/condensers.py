# Each class defines another manner of condensing responses

from openai import OpenAI
import random
import re
import json
import tiktoken

from ragatouille import RAGPretrainedModel
from transformers import pipeline
from utils.gpt_output_utils import extract_code_block_data

LLM_RERANKER_PROMPT = """
"""

class CohereReranker():
    def __init__(self, api_key):
        pass

class BARTSummarizer():
    def __init__(self, model_name="Falconsai/text_summarization"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarized(self, contents, objective):
        content_input = f"## Information about {objective} \n \n {contents}"
        summarized = self.summarizer(content_input, max_length=1000, min_length=100, do_sample=False)
        return summarized


class RagatouilleReranker():
    def __init__(self):
        self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    def rerank(self, contents, objective, output_count, combine_output=True):
        if output_count > len(contents):
            if combine_output:
                results = ["------\n".join([snippet for snippet in contents])]
                return results
            return contents
            

        results = self.RAG.rerank(query=objective, documents=contents, k=output_count)

        print("ColBERT reranker result: ", results)

        results = [(result["content"], result["result_index"]) for result in results]
        results = sorted(results, key=lambda x: x[1])
        results = [result[0] for result in results]
        
        if combine_output:
            results = ["-----\n".join([result for result in results])]

        return results


class LLMReranker():

    def __init__(self, api_key, base_url, model_name, output_count=5, process_chunk_count=10):
        self.model_name = model_name
        self.output_count = output_count
        self.process_chunk_count = process_chunk_count
        self.openai = OpenAI(api_key=api_key, base_url=base_url)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def rerank(self, contents, objective, output_count, process_chunk_count, combine_output=False): 
        # random.shuffle(contents)
        scored_snippets = []

        for range_start in range(0, len(contents), process_chunk_count):
            range_end = max(range_start + process_chunk_count, len(contents))
            contents_range = contents[range_start:range_end]

            snippets_string = "\n".join([f"\n ### Snippet {index + 1}: \n \n {contents}" for index, contents in enumerate(contents_range)])

            reranker_prompt_size = len(self.encoding.encode(LLM_RERANKER_PROMPT))
            objective_prompt_size  = len(self.encoding.encode(objective))
            snippets_prompt_size = len(self.encoding.encode(snippets_string))

            print("Sending over a request with ", reranker_prompt_size, " + ", objective_prompt_size,  " + ", snippets_prompt_size)

            response = self.openai.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": LLM_RERANKER_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Objective: {objective}"
                    },
                    {
                        "role": "user",
                        "content": f"# Snippets \n \n {snippets_string}"
                    }
                ],
                model=self.model_name,
                temperature=0.1
            )

            # Should expect a JSON response
            response = response.choices[0].message.content
            json_blocks = extract_code_block_data(response, "json")

            for block in json_blocks:
                block = json.loads(block)
                print("Block: ", block)
                for item in block:
                    snippet_id = item["snippet_id"]
                    relevance = item["relevance"]
                    
                    if type(snippet_id) is str:
                        snippet_id = int(snippet_id)
                    
                    if type(relevance) is str:
                        relevance = int(relevance)
                    
                    if snippet_id >= 0 and snippet_id < len(contents_range):
                        snippet_text = contents_range[snippet_id - 1]
                        scored_snippets.append((snippet_text, relevance))
        
        scored_snippets = sorted(scored_snippets, key=lambda x: x[1])
        scored_snippets = list(reversed(scored_snippets))

        print("Length of all scored_snippets: ", len(scored_snippets))

        scored_snippets = scored_snippets[:min(output_count, len(scored_snippets))]
        if combine_output:
            results = ["-----\n".join([snippet[0] for snippet in scored_snippets])]
            return results

        return scored_snippets[:min(output_count, len(scored_snippets))]