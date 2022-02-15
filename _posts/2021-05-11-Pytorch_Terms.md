---
title: Pytorch Terms
tags:
  - Machine Learning
  - Code
  - Neural Network
  - Pytorch
---
Code term for Pytorch
<!--more-->

### CUDA
cuda available check: ```torch.cuda.is_available()```<br>
cuda device 설정 <br>
```py
device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu')
```
Environment GPU device 설정 <br>
```py
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```
Model과 Param에 ```.cuda()``` 또는 ```.to(device)``` 설정<br>

### Dataset
---
##### DataLoader
```from torch.utils.data import DataLoader```로부터 DataLoader를 불러와서 사용 <br>
ex) ```dataloader = DataLoader(dataset)```
옵션으로는 
- batch_size=1
- shuffle=False
- num_workers=0 <br>
등이 있다. dataset에는 dataset이나 dataset을 상속받은 custom class를 넣을 수 있다.<br>
```py
 for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
```
의 형식으로 쓸 수 있으며, batch에는 dataloader의 return, i에는 반복 횟수가 들어간다. <br>

##### Dataset
```from torch.utils.data import Dataset```를 import 후 inheritance하여 사용<br>
```py
class ImageDataset(Dataset):
    def __init__(self, data_dir, opt) -> None:
        super().__init__()
        pass
    def __getitem__(self, index):
        # dataloader에서 input으로 하나씩 들어갈 data들
        # dict 형식으로 return 가능
    def __len__(self):
        # input file의 개수를 return
```
##### For Custom Dataset
- ```import torchvision.transforms as transforms```
  - ```transform.Compose()```를 통해 data전처리 Compose가능
```py
transforms.Compose([
            transforms.Resize((int(size), int(size)), Image.BICUBIC), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
```

### Model
---
```torch.nn.Module```을 보통 상속받는다.<br>
```def forward(self, x)```와 ```def backward(self)```등을 정의 할 수 있음
- ```torch.nn``
  - ```nn.Conv2d()```, ```nn.LeakyReLU()```, ```nn.ReflectionPad2d()```, ```nn.InstanceNorm2d()```, ```nn.BatchNorm2d()```, ```nn.ConvTranspose2d()```등이 define되어 있음.
  - ```nn.Sequential()```로 Model의 layer를 만들 수 있음
```py
nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2), 
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.LeakyReLU(0.2))
```
- ```nn.functional as F```
  - nn과 마찬가지로 model에 필요한 function들이 define되어 있음


### Training
---
##### Loss
Loss는 torch.nn에서 사용가능 <br>
> ex) ```torch.nn.L1Loss()```, ```torch.nn.MSELoss()```, etc... <br>
> ```torch.nn.MSELoss()```는 default가 mean이지만 reduction option을 sum으로 설정 가능
##### Optimizer
torch.optim에서 사용가능
> ex) ```torch.optim.Adam()```, ```torch.optim.SGD()```, ```torch.optim.RMSprop()```, etc...<br> 
learning rate와 옵션들을 function에 따라서 lr= ... 등으로 설정 가능 <br>
> ex) ```torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))```

##### lr_scheduler
각 parameter나 시간에 따라 leraning rate를 조절해준다. Optimizer.step()을 호출한 후 Scheduler.step()을 호출한다. <br>
(반대로하면 첫번째 learning rate를 optimizer가 건너뛰게 된다.)
- 공통
  - last_epoch: default는 -1. 처음 시작 epoch를 설정하는 option
  - verbose: default는 False, True로 설정하면 update될 때 메세지를 출력한다.
- ```torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)```
  - 설정된 lr_lambda 함수에 따라 learning rate를 조절해준다. (초기 lr에 곱할 factor를 결정하는 함수)
- ```torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)```
  - LambdaLR과 비슷하지만 초기 lr이 아니라 누적곱이라는 점이 다르다.
- ```torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)```
  - step size마다 gamma 비율로 lr을 감소시킨다.
- ```torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)```
  - step size가 아닌 lr을 감소시킬 epoch를 지정

##### grad
grad가 붙은 함수들은 보통 미분 계산에 사용된다
- ```zero_grad()```
  - Backpropagation을 사용하기 전 변화도를 0으로 만들어주는 함수
  - torch에서 backward시, autograd를 사용하게 되는데, autograd에선 grad를 합쳐주기 때문에 그 전에 gradient를 0으로 만들어 주어야한다.
  - <https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch>
- ```required_grad```
  - required_grad = True 라면, 이 Tensor에 대한 미분값을 계산하여 Tensor에 저장
  - required_grad = False 라면, Tensor의 미분값을 계산할 필요가 없다는 것
  - default는 True다. 

### Test
---
##### vidsom
loss나 PSNR등 test중에 원하는 값을 visualize할 수 있도록 도와줌<br>
```import visdom```<br>
```python -m visdom.server```를 terminal에 입력 후 url창에 http://localhost:8097
```py
vis = visdom.Visdom()

... pass ...

if i == 0:
    plot = vis.line(Y=torch.Tensor([PSNR_clear]), X=torch.Tensor([i+1]), 
        opts = dict(xlabel='i', ylabel='PSNR', ytickmin=0, ytickmax=90, ytickstep=10, xtickstep=1, title='PSRN' ,legend=['clear', 'mean']))
else:
    vis.line(Y = torch.Tensor([PSNR_clear]), X = torch.Tensor([i+1]), win=plot, update='append', name='clear')

if i+1 == len(dataloader):
            PSNR_mean /= len(dataloader)
            vis.line(Y = torch.Tensor([PSNR_mean]), X = torch.Tensor([0]), win=plot, update='append', name='mean')
```

##### tensorboard

```from torch.utils.tensorboard.write import SummaryWriter```<br>
```py
writer = SummaryWriter(log_dir=path)

... pass ...

writer.add_image(tag, image_tensor, step)
writer.add_scalar(tag, scalar_value, step)

```

##### Load Model
ex) ```model.load_state_dict(torch.load("path"))```

##### Model Save
ex) ```torch.save(model.state_dict(), 'path/name.pth')```

### etc
---
##### argparser
argument 설정에 용이. ```import argparse```로 사용<br>
```py
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=15, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
opt = parser.parse_args()
```

##### image print
make_grid, torch.cat, save_image 사용

##### nn.Sequential vs nn.ModuleList

nn.Sequential과 nn.ModuleList는 기본적으로 비슷하지만 약간의 차이점이 있습니다. <br>
nn.Sequential의 경우 내부에 저장된 nn.Module들을 연결하여 하나의 nn.Module처럼 사용할 수 있도록 해줍니다.<br>
내부적으로 forward도 수행해주기 때문에 여러 개의 nn.Module이 하나의 Module처럼 쓰이길 원할 때 묶어서 사용할 수 있습니다. <br>

```py
self.block = nn.Sequential(nn.Conv2d(in_dim, out_dim),
                      nn.GroupNorm(),
                      nn.SiLU())

def forward(self, x):
    return self.block(x)
```

nn.ModuleList는 python의 기본 list와 같이 단순히 Module들을 list로 묶은 것과 같습니다. <br>
python의 기본 list와 다른 점은 pytorch가 ModuleList내부의 nn.Module들을 인식할 수 있게 해줍니다.
ython의 기본 list로 nn.Module들을 저장하게 되면, network의 param에 내부 module들이 포함되지 않습니다. <br>
nn.Sequential과 다른점은 nn.ModuleList는 내부적으로 forward를 수행하지 않기 때문에 각 Module들끼리 연결되어 있지 않습니다. 
반복되는 구조의 Module들에 설정된 param이 다를 때, ModuleList로 묶어, for문과 함께 parameter를 설정해주는 경우에 많이 쓰일 수 있습니다.<br>
```py
self.list = nn.ModuleList([])

for i in range(num):
    self.list.append(nn.ModuleList([
                    block(dim, dim_out),
                    block(dim_out, dim_out),
                    sampling(dim_out)
                ]))

def forward(self, x):
    for block, block2, sample in self.list:
        x = block(x)
        x = block2(x)
        x = sample(x)
```

> <https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463>