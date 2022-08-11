import os
import uuid
from torch.optim import Adam
from torch.nn import BCELoss
import mlflow
import torch
import argparse
from torch.utils.data import DataLoader
from model import Model
from dataset import DS


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--mlflow_host", type=str, default="http://127.0.0.1:5000/")
parser.add_argument("--ds_path", type=str, required=True)

args = vars(parser.parse_args())

print(args)

mlflow.set_registry_uri(args["mlflow_host"])
mlflow.set_tracking_uri(args["mlflow_host"])

exp_id = os.environ.get("EXP_NAME", uuid.uuid4())
mlflow.set_experiment(f"{exp_id}")

with mlflow.start_run():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)

    epoch = args["epoch"]
    lr = args["lr"]

    mlflow.log_param("Epochs", epoch)
    mlflow.log_param("Learning Rate", lr)

    criterion = BCELoss()
    optim = Adam(model.parameters(), lr=lr)

    ds_path = args["ds_path"]
    tdl = DataLoader(DS(f"{ds_path}/Train"), batch_size=16, shuffle=True)
    vdl = DataLoader(DS(f"{ds_path}/Val"), batch_size=16, shuffle=True)

    for e in range(epoch):
        training_running_loss = 0.0
        for i, data in enumerate(tdl):
            optim.zero_grad()

            x, y = data
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            loss.backward()
            optim.step()

            training_running_loss += loss.item()

        print("Epoch", e, "  Train Loss", training_running_loss / (i + 1))


        mlflow.log_metric("Train loss", training_running_loss / (i + 1), e)

        # Validation
        with torch.no_grad():  # this stops pytorch doing computational graph
            validation_running_loss = 0.0
            correct = 0
            for i, data in enumerate(vdl):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)

                y_pred_sigmod = torch.clone(y_pred)
                y_pred_sigmod[y_pred_sigmod < 0.5] = 0
                y_pred_sigmod[y_pred_sigmod >= 0.5] = 1

                correct += (y_pred_sigmod == y).float().sum()

                loss = criterion(y_pred, y)

                validation_running_loss += loss.item()

            accuracy = correct.item() / len(vdl.dataset)
            print("Epoch", e, "  Validation Loss", validation_running_loss / (i + 1), ",  Accuracy: ", accuracy)
            mlflow.log_metric("Validation loss", validation_running_loss / (i + 1), e)
            mlflow.log_metric("Validation accuracy", accuracy, e)

    mlflow.pytorch.log_model(model, "model")



