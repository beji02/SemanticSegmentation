import torch
from UnetUtils import dice_loss
import wandb

def train(model, num_epochs, train_dataloader, test_dataloader, optimizer, loss_fn):
    train_losses = []
    test_losses = []

    columns=["image", "guess", "truth", "epoch"]
    test_table = wandb.Table(columns=columns)

    for i in range(num_epochs):
        print('EPOCH {}:'.format(i))
        
        model.train()
        train_one_epoch(model, train_dataloader, optimizer, loss_fn)
        
        model.eval()
        train_loss = 0
        test_loss = 0

        with torch.no_grad():
            for batch in train_dataloader:
                inputs, labels = batch
                outputs = model(inputs)
                train_loss += loss_fn(outputs, labels)
                    
            _id = 0
            for batch in test_dataloader:
                inputs, labels = batch
                outputs = model(inputs)
                test_loss += loss_fn(outputs, labels)
            
                if _id == 0:
                    test_table.add_data(wandb.Image(inputs[0]), wandb.Image(outputs[0]), wandb.Image(labels[0]), i)
                    _id += 1

        train_loss = train_loss / len(train_dataloader)
        test_loss = test_loss / len(test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        metrics = {"train_loss": train_loss,
                   "epoch": i,
                   "test_loss": test_loss
                   }
        wandb.log(metrics)
        print('LOSS train {} valid {}'.format(train_loss, test_loss))

    wandb.log({"test_predictions" : test_table})

    train_losses = [train_loss.item() for train_loss in train_losses]
    test_losses = [test_loss.item() for test_loss in test_losses]
    return train_losses, test_losses


def train_one_epoch(model, train_dataloader, optimizer, loss_fn):
    for i, batch in enumerate(train_dataloader):
        inputs, labels = batch
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print('batch: {} loss: {}'.format(i, loss))


def loss_fn_unet(outputs, labels):
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss += dice_loss(
                torch.nn.functional.softmax(outputs, dim=1).float(),
                labels.float(),
                multiclass=True
            )
    return loss