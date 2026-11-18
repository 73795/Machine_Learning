import os
import yaml
import mlflow
import torch

from data import get_data
from models import CNN
from train import train_model
from train import evaluate



def main():
    with open("../config.yaml") as file:
        config = yaml.safe_load(file)
    torch.manual_seed(config["seed"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    num_workers = config["data"]["num_workers"]
    checkpoint_path = config["checkpoint"]["path"]


    train_loader, val_loader, test_loader = get_data(
        batch_size,
        num_workers
    )
    model = CNN()
    start_epoch = 0




    if os.path.exists(checkpoint_path):
        answer = input(
            "continue training? (y/n): "
        )
        if answer.lower() == "y":
            checkpoint = torch.load(
                checkpoint_path
            )
            model.load_state_dict(
                checkpoint["model_state"]
            )
            start_epoch = checkpoint["epoch"] + 1
    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )
    with mlflow.start_run():
        mlflow.log_param(
            "batch_size",
            batch_size
        )
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs,
            learning_rate,
            checkpoint_path,
            start_epoch
        )
        accuracy = evaluate(
            model,
            test_loader,
            device
        )
        print(
            "Test accuracy:",
            accuracy
        )
if __name__ == "__main__":
    main()
