import torchvision.transforms as transforms
from utils import progress_bar
import sys
import torch
import torchvision
import resnet
model_path = sys.argv[1]
model = resnet.ResNet34()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight = torch.load(model_path)
if type(weight) == dict and "net" in weight.keys():
    weight = weight['net']
try:
    model.load_state_dict(weight)
except:
    model = torch.nn.DataParallel(model)
    model.load_state_dict(weight)
net = model.cuda()
net.eval()
# model = torch.load("checkpoint/"+model_file)
inDist = sys.argv[2]
outDist = sys.argv[3]
def get_D(dataName):
  return torchvision.datasets.ImageFolder("./data/kfold/{}".format(dataName), transform=transform)
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    # transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])
trainset = get_D(inDist)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = get_D(outDist)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
train_loss = 0
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
	inputs, targets = inputs.to(device), targets.to(device)
	outputs = net(inputs)
	_, predicted = outputs.max(1)
	total += targets.size(0)
	correct += predicted.eq(targets).sum().item()
	progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
