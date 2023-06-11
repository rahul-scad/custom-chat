from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from PIL import Image


class AbstractMultimodalPipeline(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        
        pass

    @staticmethod
    @abstractmethod
    def image_start() -> Optional[str]:
        
        pass

    @staticmethod
    @abstractmethod
    def image_end() -> Optional[str]:
       
        pass

    @staticmethod
    @abstractmethod
    def placeholder_token_id() -> int:
       
        pass

    @staticmethod
    @abstractmethod
    def num_image_embeds() -> int:
        
        pass

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
       
        pass

    @staticmethod
    @abstractmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        
        pass

    @staticmethod
    @abstractmethod
    def placeholder_embeddings() -> torch.Tensor:
        
        pass

    def _get_device(self, setting_name: str, params: dict):
        if params[setting_name] is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(params[setting_name])

    def _get_dtype(self, setting_name: str, params: dict):
        return torch.float32 if int(params[setting_name]) == 32 else torch.float16
