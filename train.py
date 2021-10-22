import argparse
import os
import sys
import random
from PIL import Image
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch.autograd import Variable

from models.model_factory import StyleModelFactory
from models.model_utils import FeatureExtractor
from utils import *

"""
python train.py --dataset_path C:/Users/dnn_server/PycharmProjects/fast_neural_style_transfer/images/content/dataset  --style_image C:/Users/dnn_server/PycharmProjects/fast_neural_style_transfer/images/styles/mosaic.jpg

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to training dataset")
    parser.add_argument("--style_image", type=str, default="images/styles/mosaic.jpg", help="path to style image")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=640, help="Size of training images")
    parser.add_argument("--style_size", type=int, help="Size of style image")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_model", type=str, default=None, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=10, help="Batches between saving image samples")
    args = parser.parse_args()

    style_name = args.style_image.split("/")[-1].split(".")[0]
    os.makedirs(f"images/outputs/{style_name}-training", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloader for the training data
    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Defines networks
    style_transformer = StyleModelFactory.StyleResnet18(args.checkpoint_model)
    feature_extractor_vgg = FeatureExtractor(args.style_image, args.batch_size)


    # Define optimizer
    optimizer = Adam(style_transformer.model.parameters(), args.lr)

    # Sample 8 images for visual evaluation of the model
    image_samples = []
    for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.jpg"), 8):
        image_samples += [style_transform(args.image_size)(Image.open(path))]
    image_samples = torch.stack(image_samples)

    def save_single_image(batches_done):
        style_transformer.model.eval()
        # Prepare input
        transform = style_transform()
        single_img_path = r"C:\Users\dnn_server\PycharmProjects\fast_neural_style_transfer\images\content\dataset\1\360.jpg"

        image_tensor = Variable(transform(Image.open(single_img_path))).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(style_transformer.model(image_tensor)).cpu()

        save_image(stylized_image, f"images/outputs/{style_name}-training/{batches_done}_sample.jpg")
        style_transformer.model.train()

    def save_sample(batches_done):
        """ Evaluates the model and saves image samples """
        style_transformer.model.eval()
        with torch.no_grad():
            output = style_transformer.model(image_samples.to(device))
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
        style_transformer.model.train()

    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = style_transformer.model(images_original)

            content_loss = feature_extractor_vgg.get_content_loss(images_original, images_transformed, args.lambda_content)
            style_loss = feature_extractor_vgg.get_style_loss(images_transformed, args.lambda_style)

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    len(train_dataset),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            batches_done = epoch * len(dataloader) + batch_i + 1
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)
                save_single_image(batches_done)

            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                checkpoint_path = "checkpoints/{}_{}.pth".format(style_name, batches_done)
                style_transformer.save_checkpoint(checkpoint_path)
