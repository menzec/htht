import torch
import torchvision
import datetime
import Dataset
import argparse
import torch.nn as nn

BATCH_SIZE = 4
LEARNING_RATE = 0.000001
ALL_EPOCH = 150
MILESTONES = [60, 110]
GAMMA = 0.1


def arg_parse():
    parser = argparse.ArgumentParser(
        description="A simple test of Resnet on UCMerced.")
    parser.add_argument(
        '-d',
        '--datadir',
        help="data folder directory",
        default=r"D:\data\UCM-full",
    )
    parser.add_argument(
        '-e', '--epoch', default=ALL_EPOCH, type=int, help="total epoch")
    args = parser.parse_args()
    return args


def net_train(net, epoch, para, data):
    device, optimizer, scheduler = para
    criterion = nn.CrossEntropyLoss()
    trainloader = data
    torch.no_grad()
    scheduler.step()
    print('Epoch: %d' % (epoch + 1))
    startime = datetime.datetime.now()
    print('startime: %s' % startime)
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    net = net.to(device)
    trainlog = open("./trainlog.txt", "a")
    for i, data in enumerate(trainloader, 0):
        # 准备数据
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 每训练  个batch打印一次loss和准确率
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print(
            '[epoch:%03d, iter:%05d] Loss: %.03f | Acc: %.3f%%' %
            (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
             100.0 * float(correct) / float(total)),
            file=trainlog)
        trainlog.flush()
        if (i + 1) % 10 == 0:
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%%' %
                  (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                   100.0 * float(correct) / float(total)))
        inputs, labels = inputs.to(torch.device('cpu')), labels.to(
            torch.device('cpu'))
    net = net.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    endtime = datetime.datetime.now()
    print("scheduler-lr:", scheduler.optimizer.param_groups[0]["lr"])
    print(
        "scheduler-lr:  ",
        scheduler.optimizer.param_groups[0]["lr"],
        file=trainlog)
    print('endtime: %s |cost time: %s' % (endtime, endtime - startime))
    torch.save(net.state_dict(), './modelpara/net_%03d.pth' % (epoch + 1))
    print(
        'endtime: %s |cost time: %s' % (endtime, endtime - startime),
        file=trainlog)
    trainlog.close()
    print('Saving model net_%03d.pth' % (epoch + 1))


def net_test(net, data, device, epoch, classes_label):
    testloader = data
    print('Testing...%d' % (epoch + 1))
    torch.no_grad()
    correct = 0
    total = 0
    class_acc = [[0, 0] for i in range(len(classes_label))]
    net = net.to(device)
    for data in testloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_info = (predicted == labels)
        correct += correct_info.sum()
        images, labels = images.to(torch.device('cpu')), labels.to(
            torch.device('cpu'))
        for i in range(len(correct_info)):
            class_acc[labels[i]][0] += int(correct_info[i])
            class_acc[labels[i]][1] += 1
    print('测试分类准确率为：%.3f%%' % (100.0 * float(correct) / float(total)))
    torch.cuda.empty_cache()
    print("epoch %3d Test Finished" % (epoch + 1))
    testlog = open("./testlog.txt", "a")
    print(
        'Epoch:%3d,测试分类准确率为：%.3f%%' % (epoch + 1,
                                       100.0 * float(correct) / float(total)),
        file=testlog)
    testlog.close()
    with open("./test_detail_log.txt", "a", encoding="UTF-8") as test_de_log:
        test_de_log.write('Epoch:%3d,测试分类准确率为：%.3f%%\n' %
                          (epoch + 1, 100.0 * float(correct) / float(total)))
        for i, cl_acc in enumerate(class_acc):
            test_de_log.write(
                "  %3d: %%%6.2f--%s\n" % (i, 100 * cl_acc[0] / cl_acc[1],
                                          classes_label[i]))


def main():
    # 定义是否使用GPU
    args = arg_parse()
    datadir = args.datadir
    all_epoch = args.epoch
    if all_epoch < 1:
        all_epoch = ALL_EPOCH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    dataname = "UCMerced"
    model_name = "ResNet50"
    classes_label = Dataset.UCMerced.label()
    traindata, testdata = Dataset.UCMerced(datadir,
                                           BATCH_SIZE).getdata(pretrained=True)
    net = torchvision.models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, len(classes_label))
    net.load_state_dict(torch.load("./net_071.pth"))
    # for param in net.parameters():
    #     if len(param.
    #            shape) == 2 and param.shape[0] == 21 and param.shape[1] == 2048:
    #         break
    #     param.requires_grad = False
    # # net.fc = nn.Linear(net.fc.in_features, len(classes_label))
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA)
    para = [device, optimizer, scheduler]
    startime = datetime.datetime.now()
    print('Start %s + %s,startime:%s' % (model_name, dataname, startime))
    for epoch in range(0, ALL_EPOCH):
        net_train(net, epoch, para, traindata)
        net_test(net, testdata, device, epoch, classes_label)
    endtime = datetime.datetime.now()
    print('%s + %s Finished! Endtime:%s, costtime:%s' %
          (model_name, dataname, endtime, endtime - startime))


if __name__ == '__main__':
    main()