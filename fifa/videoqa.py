from abc import ABC, abstractmethod

class BaseVQAGenerator(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def answer(self, question: str, video_path: str) -> str:
        pass

