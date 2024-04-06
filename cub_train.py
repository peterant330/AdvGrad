from datasets import CUB_train, CUB_test
import torch
import torch.nn as nn
import os
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epsilon', metavar='N', type=float,
                        help='an integer for the accumulator')
    parser.add_argument('--data_dir', type=str, help='address of the data')
    parser.add_argument(
        "--snapshot_path",
        default="checkpoint_cub/",
        type=str,
    )
    args = parser.parse_args()
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    train_data = CUB_train(root_dir=args.data_dir, csv_file="train.csv")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=16, shuffle=True, num_workers=4)
    val_data = CUB_test(root_dir=args.data_dir, csv_file="test.csv")
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=8, shuffle=False, num_workers=4)

    net = EfficientNet.from_name('efficientnet-b0', num_classes=200).cuda()
    optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.999, 0.999), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    epsilon = args.epsilon
    for epoch in range(200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        org_grad = 0
        for idx, (img, seg, label) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            seg = (0.5 - seg)
            seg = seg.unsqueeze(1).cuda()

            if epsilon > 0:
                org = torch.tensor(img, requires_grad=True)
                out = net(org)
                loss = criterion(out, label)
                loss.backward()
                org_grad = org.grad.data
                org_grad = org_grad / torch.norm(org_grad, p=2, dim=[1,2,3], keepdim=True)
                img += 2 * epsilon * org_grad * seg
            outputs = net(img)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        print('Epoch: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (idx + 1), 100. * correct / total, correct, total))
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (img, seg, label) in enumerate(val_loader):
                img = img.cuda()
                label = label.cuda()
                outputs = net(img)
                loss = criterion(outputs, label)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
            print('Epoch: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' %  (epoch, test_loss / (idx + 1), 100. * correct / total, correct, total))

        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir({args.snapshot_path}):
                os.mkdir({args.snapshot_path})
            torch.save(state, os.path.join(args.snapshot_path, 'cub_{}.pth'.format(epsilon)))
            best_acc = acc
