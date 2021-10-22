import torch
from collections import namedtuple

from PIL import Image
from torchvision import models

from utils import style_transform


class SlicedVGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(SlicedVGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class FeatureExtractor():
    def __init__(self, style_img_path: str, batch_size: int):
        self._style_img_path = style_img_path
        self._batch_size = batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fe = SlicedVGG16(requires_grad=False).to(self._device)
        self._l2_loss = torch.nn.MSELoss().to(self._device)

        self._gram_style = self._load_style()

    def get_content_loss(self, images_original, images_transformed, content_weight: float):
        # Extract features
        features_original = self._fe(images_original)
        features_transformed = self._fe(images_transformed)

        # Compute content loss as MSE between features
        content_loss = content_weight * self._l2_loss(features_transformed.relu2_2, features_original.relu2_2)
        return content_loss

    def get_style_loss(self, images_transformed, style_weight: float):
        style_loss = 0
        features_transformed = self._fe(images_transformed)

        for ft_y, gm_s in zip(features_transformed, self._gram_style):
            gm_y = self._gram_matrix(ft_y)
            style_loss += self._l2_loss(gm_y, gm_s[: self._batch_size, :, :]) # images.size(0)
        style_loss *= style_weight
        return style_loss

    def _load_style(self):
        # Load style image
        style = style_transform()(Image.open(self._style_img_path))
        style = style.repeat(self._batch_size, 1, 1, 1).to(self._device)

        # Extract style features
        features_style = self._fe(style)
        gram_style = [self._gram_matrix(y) for y in features_style]
        return gram_style

    def _gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

















