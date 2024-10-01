from abc import ABC, abstractmethod
from llmlingua import PromptCompressor

class Compressor(ABC):
    @abstractmethod
    def compress(self, text: str) -> str:
        pass

class LLMLinguaCompressor(Compressor):
    def __init__(self, use_small: bool = False, compress_rate = 0.33):
        self.llm_lingua = PromptCompressor(
            model_name=("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank" if use_small else "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"),
            use_llmlingua2=True
        )
        self.compress_rate = compress_rate
        
    def compress(self, text: str) -> str:
        return self.llm_lingua.compress_prompt(text, rate=self.compress_rate)
    
class CommonKeywordCompressor(Compressor):
    def __init__(self, keywords=["class", "def", "if", "else", "for", "in"]):
        self.keywords = keywords

    def compress(self, text: str) -> str:
        return " ".join([word for word in text.split() if word not in self.keywords])