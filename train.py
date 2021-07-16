import os
from datetime import datetime

import torch

import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter


def train(dataloader, model, loss_fn, optimizer, scheduler, arg, validation):
    """
    trains the given model on the given datset for the number of epochs in arg.epoch
    using given loss function, optimizer and scheduler
    every 5 epochs the state_dict is saved in trained_models/ with the name of the model, dataset and number of epoch
    :param dataloader: gibt aus Input, Ground_Truth und ggf. Segmentierungsmaske
    :param model:
    :param loss_fn:
    :param optimizer:
    :param scheduler:
    :param arg: used: arg.epoch arg.dataset_name for psm specific settings, arg.model_name for saving pth
    :param validation: dataloader with validation data, if used
    :return:
    """

    # tensorboard initializing
    log = "testValidationLog"
    writer_input = SummaryWriter(f"logs/{arg.model_name}{log}/input")
    writer_gt = SummaryWriter(f"logs/{arg.model_name}{log}/gt")
    writer_output = SummaryWriter(f"logs/{arg.model_name}{log}/output")
    dif = True if 'dif' in arg.dataset_name else False
    tb = next(iter(dataloader))
    if dif:
        writer_result = SummaryWriter(f"logs/{arg.model_name}{log}/result")
    if validation is not None:
        writer_validation = SummaryWriter(f"logs/{arg.model_name}{log}/validation")
        val_step = 0
    step = 0

    # model to cuda + make save_path for pths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    save_path = 'trained_models/test/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epoch_loss = []

    for epoch in range(arg.epoch):
        print(f"Epoch: {epoch}")
        print("--------------")
        size = len(dataloader.dataset)
        for batch, load in enumerate(dataloader):
            model.train()
            # forward propagation
            pred, gt = forward(model, load, arg.dataset_name, device)
            loss = loss_fn(pred, gt)
            epoch_loss.append(loss.item())
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # some feedback+tensorboard output every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(load[0])
                print(f"{datetime.now()} Epoch:{epoch}/{arg.epoch}, Step: {current:>5d}/{size:>5d}, loss: {loss:>7f}")
                # writer_result = writer_result if dif else None
                # tensorboard_feedback(tb, model, writer_input, writer_gt, writer_output, arg.dataset_name, device, step, writer_result)
                step += 1

        # save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       save_path+ f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_' + arg.model_name + '_on_' + arg.dataset_name + str(epoch + 1) + '.pth')

        if validation is not None:
            model.eval()
            val_loss = []
            for batch, load in enumerate(validation):
                with torch.no_grad():
                    pred, gt = forward(model, load, arg.dataset_name, device)
                    val_loss.append(loss_fn(pred.cpu(), gt.cpu()).item())
            writer_validation.add_scalar('Average Batch Validation loss', sum(val_loss)/len(val_loss), global_step=val_step)
            writer_validation.add_scalar('Average Batch Training loss', sum(epoch_loss)/len(epoch_loss), global_step=val_step)
            val_step+=1

        scheduler.step()


def visualize(loader, model):
    figure = plt.figure(figsize=(8, 8))
    for x, test_labels in loader:
        x = x.to(device='cuda')

        with torch.no_grad():
            output = model(x)
        cols, rows = 2, 3
        for i in range(1, rows + 1):
            # pred = torch.clamp(output[i], 0, 1)
            pred = output[i].cpu().detach()
            figure.add_subplot(rows, cols, 2 * i - 1)
            plt.axis("off")
            plt.imshow(pred.permute(1, 2, 0), vmin=0, vmax=1)
            figure.add_subplot(rows, cols, 2 * i)
            plt.axis("off")
            plt.imshow(test_labels[i].permute(1, 2, 0))
        plt.show()
        break
def forward(model, load, name, device):
    input, gt = load[0].to(device=device), load[1].to(device=device)
    if 'dif' in name: gt = gt - input  # prepare ground truth for dif if necessary
    pred = model(input)
    # use segmentation on prediction+ ground truth
    if 'psm' in name:
        mask = load[2].to(device=device)
        pred = pred * mask
        gt = gt * mask
    return pred, gt

def tensorboard_feedback(tb, model, writer_input, writer_gt, writer_output, name, device, step, writer_result=None):
    tb_output, tb_gt = forward(model, tb, name, device)
    tb_input = tb[0].to(device=device)
    img_grid_input = torchvision.utils.make_grid(tb_input.reshape(-1, 3, 128, 128), normalize=True)
    img_grid_gt = torchvision.utils.make_grid(tb_gt.reshape(-1, 3, 128, 128), normalize=True)
    img_grid_output = torchvision.utils.make_grid(tb_output.reshape(-1, 3, 128, 128), normalize=True)

    writer_input.add_image("Input Images", img_grid_input, global_step=step)
    writer_gt.add_image("Ground Truth", img_grid_gt, global_step=step)
    writer_output.add_image("Output Images", img_grid_output, global_step=step)
    if writer_result is not None:
        img_grid_res = torchvision.utils.make_grid((tb_output + tb_input).reshape(-1, 3, 128, 128),
                                                   normalize=True)
        writer_result.add_image("Result Images", img_grid_res, global_step=step)