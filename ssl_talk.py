# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from utils import imshow, ImageNet32 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Torch device: {device}')
torch.set_float32_matmul_precision('high')  # TF32 on Ampere+; no-op elsewhere

# %%
# simple shared normalization for all datasets, close enough to the true
# per-channel stats of both CIFAR-10 and ImageNet32
NORM_MEAN, NORM_STD = 0.5, 0.25

def get_cifar10_data():
    # Data augmentation transformations. Not for Testing!
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar/', transform=augment, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,
                                              num_workers=12, persistent_workers=True, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data/cifar/', train=False, transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
                                             num_workers=4, persistent_workers=True, pin_memory=True)

    return {'train': trainloader, 'test': testloader, 'classes': trainset.classes}

cifar10 = get_cifar10_data()

# %%

def visualize_data(dataloader, classes=None):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images[:8]
    print(images.size()) # N C H W

    # print labels
    print("Labels:" + ', '.join('%9s' % labels[j].item() for j in range(8)))
    if classes is not None:
        print("Classes:" + ', '.join('%s' % classes[labels[j]] for j in range(8)))

    # show images
    grid = torchvision.utils.make_grid(images)
    grid = grid * NORM_STD + NORM_MEAN
    imshow(grid)

# visualize_data(cifar10['train'], cifar10['classes'])


# %%
class Block(nn.Module):
    """One transformer block: attention + feed-forward, both with residuals."""
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Tokenizer(nn.Module):
    def __init__(self, patch_size, dim, num_patches):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))  # learned

    def forward(self, x):                   # (B, 3, 32, 32)
        x = self.patch_embed(x)             # (B, 128, 8, 8)
        x = x.flatten(2).transpose(1, 2)    # (B, 64, 128)
        # Add in class token + positional embeddings
        cls_tokens = self.cls_token.expand(len(x), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) + self.pos_embed  # (B, 65, 128)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 64
        self.tokenizer = Tokenizer(patch_size, dim, num_patches)
        self.encoder = Encoder(dim, depth, heads, mlp_dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):                               # (B, 3, 32, 32)
        x = self.tokenizer(x)                           # (B, 65, 128)
        x = self.encoder(x)
        return self.classifier(x[:, 0])                 # classify from the CLS token

# fixed seed before each model so comparable models start from identical
# weights, random seed after so the training runs stay randomized
torch.manual_seed(1234)
vit_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
torch.seed()

# %%
from torchinfo import summary
summary(vit_cifar10)

# %%
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def accuracy(model, dataloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    return correct/total


def train_classification(model, train_loader, val_loader, epochs, lr=0.002):
    model.to(device)
    model.train()
    total_steps = epochs*len(train_loader)
    warmup = 2000
    step=0
    loss_ema = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler=SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

    use_amp = device.type == 'cuda'
    if use_amp:
        model = torch.compile(model)

    print("Training")
    for e in range(epochs):
        for data, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(data.to(device, non_blocking=True))
                loss = criterion(output, labels.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = .1*loss.item() + .9*loss_ema

            if(step % 100 == 0):
                print(f'{step}: {loss_ema}')
        print(f'Epoch {e} done, val accuracy: {accuracy(model, val_loader)}')
        model.train()


# %%
train_classification(vit_cifar10, cifar10['train'], cifar10['test'], 30)

# %%

def get_imagenet32_data(root='./data/imagenet32/'):
    # Data augmentation transformations. Not for Testing!
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    trainset = ImageNet32(root, 'train', transform=augment)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,
                                              num_workers=12, persistent_workers=True, pin_memory=True)

    val_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    valset = ImageNet32(root, 'val', transform=val_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False,
                                            num_workers=4, persistent_workers=True, pin_memory=True)

    classes = open(root + 'classes.txt').read().splitlines()

    return {'train': trainloader, 'test': valloader, 'classes': classes}

# %%

imagenet32 = get_imagenet32_data()
# visualize_data(imagenet32['train'], imagenet32['classes'])
torch.manual_seed(1234)
vit_imagenet32 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=1000)
torch.seed()
train_classification(vit_imagenet32, imagenet32['train'], imagenet32['test'], 5)

vit_inet_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
vit_inet_cifar10.tokenizer = vit_imagenet32.tokenizer
vit_inet_cifar10.encoder = vit_imagenet32.encoder
train_classification(vit_inet_cifar10, cifar10['train'], cifar10['test'], 30)



# %%
class Detokenizer(nn.Module):
    """Inverse of Tokenizer: drop the class token, map each patch token back to pixels."""
    def __init__(self, patch_size, dim):
        super().__init__()
        # kernel = stride = patch size, so this is a per-token linear to 3*p*p pixels
        self.patch_unembed = nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                   # (B, 65, dim)
        x = x[:, 1:]                        # throw out the class token
        side = int(x.shape[1] ** 0.5)       # 8 patches per side
        x = x.transpose(1, 2).reshape(len(x), -1, side, side)  # (B, dim, 8, 8)
        return self.patch_unembed(x)        # (B, 3, 32, 32)

# %%

class ViTAutoEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, z_dim):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 64
        self.tokenizer = Tokenizer(patch_size, dim, num_patches)
        self.encoder = Encoder(dim, depth, heads, mlp_dim)

        # squeeze every token down to z_dim and back up
        self.bottleneck = nn.Sequential(
                            nn.Linear(dim, z_dim),
                            nn.Linear(z_dim, dim))

        self.decoder = Encoder(dim, 2, heads, mlp_dim)
        self.detokenizer = Detokenizer(patch_size, dim)

    def forward(self, x):                               # (B, 3, 32, 32)
        x = self.tokenizer(x)                           # (B, 65, 128)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = x + self.tokenizer.pos_embed                # tell the decoder where each token is
        x = self.decoder(x)
        return self.detokenizer(x)                      # (B, 3, 32, 32)

# %%

def train_autoencoder(model, train_loader, epochs=1, lr=0.002, warmup=2000):
    model.to(device)
    model.train()
    total_steps = epochs*len(train_loader)
    step=0
    loss_ema = None

    criterion = nn.SmoothL1Loss(beta=.01)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler=SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

    use_amp = device.type == 'cuda'
    if use_amp:
        model = torch.compile(model)

    print("Training")
    for e in range(epochs):
        for data, _ in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(data)
                loss = criterion(output, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = .1*loss.item() + .9*loss_ema

            if(step % 100 == 0):
                print(f'{step}: {loss_ema}')
        print(f'Epoch {e} done')

# %%

torch.manual_seed(1234)
vit_ae = ViTAutoEncoder(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, z_dim=8)
torch.seed()
train_autoencoder(vit_ae, imagenet32['train'], 5)

vit_ae_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
vit_ae_cifar10.tokenizer = vit_ae.tokenizer
vit_ae_cifar10.encoder = vit_ae.encoder
train_classification(vit_ae_cifar10, cifar10['train'], cifar10['test'], 30)

# %%

class ViTMaskedAutoEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 64
        self.tokenizer = Tokenizer(patch_size, dim, num_patches)
        self.encoder = Encoder(dim, depth, heads, mlp_dim)
        self.decoder = Encoder(dim, 2, heads, mlp_dim)
        self.detokenizer = Detokenizer(patch_size, dim)

    def forward(self, image, mask):                     # (B, 3, 32, 32)
        x = self.tokenizer(image)                       # (B, 65, 128)
        encoded = self.encoder(x[:, mask])

        # masked tokens come back as zeros, pos_embed will be added in
        full = torch.zeros_like(x)
        full[:, mask] = encoded

        x = full + self.tokenizer.pos_embed             # tell the decoder where each token is
        x = self.decoder(x)
        return self.detokenizer(x)                      # (B, 3, 32, 32)

# %%

def train_masked_autoencoder(model, train_loader, epochs=1, lr=0.002, warmup=2000, mask_ratio=0.5):
    model.to(device)
    model.train()
    total_steps = epochs*len(train_loader)
    n_tokens = model.tokenizer.pos_embed.shape[1]  # 65
    f_side = int((n_tokens-1)**0.5)
    patch_size = model.tokenizer.patch_embed.kernel_size[0]  # pixels per patch side, 4
    step=0
    loss_ema = None

    criterion = nn.SmoothL1Loss(beta=.01, reduction='none')  # per-pixel, masked below
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler=SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

    use_amp = device.type == 'cuda'
    if use_amp:
        model = torch.compile(model)

    print("Training")
    for e in range(epochs):
        for data, _ in train_loader:
            data = data.to(device, non_blocking=True)
            # 1: visible, 0: not visible
            mask = torch.rand(n_tokens, device=device) > mask_ratio

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(data, mask)
                loss = criterion(output, data)          # per-pixel loss (B, 3, 32, 32)
                
                # ignore visible pixels in the loss
                visible = mask[1:].view(f_side, f_side)
                visible = visible.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
                loss[:, :, visible] = 0
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = .1*loss.item() + .9*loss_ema

            if(step % 100 == 0):
                print(f'{step}: {loss_ema}')
        print(f'Epoch {e} done')

# %%

torch.manual_seed(1234)
vit_mae = ViTMaskedAutoEncoder(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512)
torch.seed()
train_masked_autoencoder(vit_mae, imagenet32['train'], 5, mask_ratio=0.5)

vit_mae_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
vit_mae_cifar10.tokenizer = vit_mae.tokenizer
vit_mae_cifar10.encoder = vit_mae.encoder
train_classification(vit_mae_cifar10, cifar10['train'], cifar10['test'], 30)

# %%

class ViTMaskedEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 64
        self.tokenizer = Tokenizer(patch_size, dim, num_patches)
        self.encoder = Encoder(dim, depth, heads, mlp_dim)

    def forward(self, image, mask):                     # (B, 3, 32, 32)
        x = self.tokenizer(image)                       # (B, 65, 128)
        return self.encoder(x[:, mask])

# %%

def train_contrastive(model, train_loader, epochs=1, lr=0.002, warmup=2000, mask_ratio=0.5, temperature=0.1):
    model.to(device)
    model.train()
    total_steps = epochs*len(train_loader)
    n_tokens = model.tokenizer.pos_embed.shape[1]  # 65
    step=0
    loss_ema = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler=SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

    use_amp = device.type == 'cuda'
    if use_amp:
        model = torch.compile(model)

    print("Training")
    for e in range(epochs):
        for data, _ in train_loader:
            data = data.to(device, non_blocking=True)
            # 1: visible, 0: not visible
            mask_a = torch.rand(n_tokens, device=device) > mask_ratio
            mask_b = torch.rand(n_tokens, device=device) > mask_ratio
            mask_a[0] = mask_b[0] = 1

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output_a = model(data, mask_a)[:, 0] # just the cls token
                output_b = model(data, mask_b)[:, 0]

                logits = F.normalize(output_a, dim=1) @ F.normalize(output_b, dim=1).T / temperature
                labels = torch.arange(len(data), device=device)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = .1*loss.item() + .9*loss_ema

            if(step % 100 == 0):
                print(f'{step}: {loss_ema}')
        print(f'Epoch {e} done')

# %%

torch.manual_seed(1234)
vit_contrastive = ViTMaskedEncoder(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512)
torch.seed()
train_contrastive(vit_contrastive, imagenet32['train'], 5, mask_ratio=0.75)

vit_contrastive_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
vit_contrastive_cifar10.tokenizer = vit_contrastive.tokenizer
vit_contrastive_cifar10.encoder = vit_contrastive.encoder
train_classification(vit_contrastive_cifar10, cifar10['train'], cifar10['test'], 30)

# %%

class ViTLatentMIM(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 64
        self.tokenizer = Tokenizer(patch_size, dim, num_patches)
        self.encoder = Encoder(dim, depth, heads, mlp_dim)
        self.decoder = Encoder(dim, 2, heads, mlp_dim)
        self.target_encoder = copy.deepcopy(self.tokenizer.patch_embed)

    def forward(self, image, mask):                     # (B, 3, 32, 32)
        x = self.tokenizer(image)                       # (B, 65, 128)
        encoded = self.encoder(x[:, mask])

        # masked tokens come back as zeros, pos_embed will be added in
        full = torch.zeros_like(x)
        full[:, mask] = encoded

        x = full + self.tokenizer.pos_embed             # tell the decoder where each token is
        return self.decoder(x)

    def get_targets(self, x):
        with torch.no_grad():
            x = self.target_encoder(x)      # (B, 128, 8, 8) EMA patch embeddings
            x = x.flatten(2).transpose(1, 2)  # (B, 64, 128)
            return x

    def update_target_encoder(self, a=.995):
        with torch.no_grad():
            self.target_encoder.weight.mul_(a).add_(self.tokenizer.patch_embed.weight, alpha=1-a)
            self.target_encoder.bias.mul_(a).add_(self.tokenizer.patch_embed.bias, alpha=1-a)
        
# %%

def train_latent_mim(model, train_loader, epochs=1, lr=0.002, warmup=2000, mask_ratio=0.5, temperature=0.1):
    model.to(device)
    model.train()
    total_steps = epochs*len(train_loader)
    n_tokens = model.tokenizer.pos_embed.shape[1]  # 65
    step=0
    loss_ema = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler=SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

    use_amp = device.type == 'cuda'

    print("Training")
    for e in range(epochs):
        for data, _ in train_loader:
            data = data.to(device, non_blocking=True)

            # 1: visible, 0: not visible
            mask = torch.rand(n_tokens, device=device) > mask_ratio
            mask[0] = 1                                 # need the cls token

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(data, mask)              # (B, 65, 128) decoded tokens
                targets = model.get_targets(data)       # (B, 64, 128) EMA patch embeddings

                # each masked token must pick its own patch's target embedding
                # out of all 64 patches of its image; visible tokens don't count
                hidden = ~mask[1:]                      # the patch slots the model had to guess
                logits = F.normalize(output[:, 1:][:, hidden], dim=2) @ F.normalize(targets, dim=2).mT / temperature  # (B, n_hidden, 64)
                labels = hidden.nonzero().squeeze(1).expand(len(data), -1)  # each masked token's own patch index
                loss = criterion(logits.transpose(1, 2), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.update_target_encoder()

            step += 1

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = .1*loss.item() + .9*loss_ema

            if(step % 100 == 0):
                print(f'{step}: {loss_ema}')
        print(f'Epoch {e} done')

# %%

torch.manual_seed(1234)
vit_latentmim = ViTLatentMIM(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512)
torch.seed()
train_latent_mim(vit_latentmim, imagenet32['train'], 5, mask_ratio=0.75)

vit_latentmim_cifar10 = ViT(image_size=32, patch_size=4, dim=128, depth=5, heads=8, mlp_dim=512, num_classes=10)
vit_latentmim_cifar10.tokenizer = vit_latentmim.tokenizer
vit_latentmim_cifar10.encoder = vit_latentmim.encoder
train_classification(vit_latentmim_cifar10, cifar10['train'], cifar10['test'], 30)




# %%

models = {'supervised': vit_cifar10, 'imagenet32': vit_inet_cifar10,
          'autoencoder': vit_ae_cifar10, 'masked autoencoder': vit_mae_cifar10,
          'contrastive': vit_contrastive_cifar10, 'latent mim': vit_latentmim_cifar10}
for name, m in models.items():
    print(f'{name}: {accuracy(m, cifar10["test"]):.4f}')

