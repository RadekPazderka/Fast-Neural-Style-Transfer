from typing import Optional

from models.style_resnet18 import StyleResnet18Model


class StyleModelFactory():

    @staticmethod
    def StyleResnet18(checkpoint_path: Optional[str]):
        return StyleResnet18Model(checkpoint_path)