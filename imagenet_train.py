from datasets import ImageNette_train, ImageNette_test
import torch
import torch.nn as nn
import os
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', default="standard", type=str)
    parser.add_argument('--training', default="fast", type=str)
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--epsilon2', default=0, type=float)
    parser.add_argument('--data_dir', type=str, help='address of the data')
    parser.add_argument(
        "--snapshot_path",
        default="checkpoint/",
        type=str,
    )
    args = parser.parse_args()

    train_data = ImageNette_train(root_dir=args.data_dir, csv_file="train.csv")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=4)

    val_data = ImageNette_test(root_dir=args.data_dir, csv_file="val.csv")
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=8, shuffle=False, num_workers=4)

    net = EfficientNet.from_name('efficientnet-b0', num_classes=10).cuda()
    optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.999, 0.999), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    epsilon = args.epsilon
    for epoch in range(200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for idx, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            if args.mode=="standard":
                x_adv = img.detach()
            elif args.mode == 'l1':
                if args.training == 'fast':
                    x_adv = img.detach().clone().requires_grad_(True)
                    out = net(x_adv)
                    loss = criterion(out, label) * img.shape[0]
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = x_adv + args.epsilon * grad.sign()
                elif args.training == 'iterative':
                    x_adv = img.detach().clone().requires_grad_(True)
                    for t in range(7):
                        x_adv = x_adv.detach().clone().requires_grad_(True)
                        out = net(x_adv)
                        loss = criterion(out, label) * img.shape[0]
                        grad = torch.autograd.grad(loss, x_adv)[0]
                        x_adv = x_adv + 0.3 * args.epsilon * grad.sign()
                        x_adv = img + (x_adv - img).clamp(-args.epsilon, args.epsilon)
            elif args.mode == 'elastic':
                if args.training == 'fast':
                    x_adv = img.detach().clone().requires_grad_(True)
                    out = net(x_adv)
                    loss = criterion(out, label) * img.shape[0]
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = img + args.epsilon * grad.sign() + 2 *args.epsilon2 * grad
                elif args.training == 'iterative':
                    x_adv = img.detach().clone().requires_grad_(True)
                    for t in range(7):
                        x_adv = x_adv.detach().clone().requires_grad_(True)
                        out = net(x_adv)
                        loss = criterion(out, label) * img.shape[0]
                        loss = loss - \
                             1 / (4 * args.epsilon2) * torch.square(torch.norm(torch.relu(torch.abs(x_adv - img)-args.epsilon),
                                                                        p=2))
                        grad = torch.autograd.grad(loss, x_adv)[0]
                        norm = torch.linalg.norm(grad.detach().flatten(1), ord=2, dim=1)[:, None, None, None]
                        grad = grad / norm
                        x_adv = x_adv + 0.3 * grad
            elif args.mode == 'group':
                if args.training == 'fast':
                    org = torch.tensor(img, requires_grad=True)
                    out = net(org)
                    loss_0 = criterion(out, label)
                    org_grad = torch.autograd.grad(loss_0, org, retain_graph=True)[0]
                    org_grad_square = org_grad ** 2
                    grad_norm = org_grad_square.reshape(-1, 3, 16, 14, 16, 14).permute(0, 1, 2, 4, 3, 5) \
                        .flatten(4, 5).sum(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 14, 14)
                    grad_norm = torch.sqrt(grad_norm).permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, 224, 224)
                    org_grad = org_grad / grad_norm
                    x_adv = img + epsilon * org_grad.detach()
                elif args.training == 'iterative':
                    delta = torch.zeros_like(img)
                    for step in range(7):
                        delta = delta.detach().requires_grad_(True)
                        out = net(img + delta)
                        loss_0 = criterion(out, label)
                        org_grad = torch.autograd.grad(loss_0, delta, retain_graph=True)[0]
                        norm = torch.linalg.norm(org_grad.detach().flatten(1), ord=2, dim=1)[:, None, None, None]
                        org_grad = org_grad / norm
                        delta = delta + org_grad
                        delta_square = delta ** 2
                        grad_norm = delta_square.reshape(-1, 3, 16, 14, 16, 14).permute(0, 1, 2, 4, 3, 5) \
                            .flatten(4, 5).sum(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 14, 14)
                        grad_norm = torch.sqrt(grad_norm).permute(0, 1, 2, 4, 3, 5).reshape(-1, 3, 224, 224)
                        ratio = torch.where(grad_norm > epsilon, epsilon / grad_norm, 1.).cuda()
                        delta = delta.detach() * ratio
                    x_adv = img + delta
            optimizer.zero_grad()
            x_adv = x_adv.detach()
            outputs = net(x_adv)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        print(
            'Epoch: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (idx + 1), 100. * correct / total, correct, total))
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (img, label) in enumerate(val_loader):
                img = img.cuda()
                label = label.cuda()
                outputs = net(img)
                loss = criterion(outputs, label)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
            print('Epoch: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss / (idx + 1), 100. * correct / total, correct, total))

        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.snapshot_path):
                os.mkdir(args.snapshot_path)
            torch.save(state, f'./{args.snapshot_path}/imagenet_{args.mode}_{args.epsilon}_{args.epsilon2}.pth')
            best_acc = acc
