# training.py
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import HeadGearDataset
from utils.DCGAN import Generator, Discriminator
from utils.config import load_config
from utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# Initialize the SummaryWriter
writer = SummaryWriter()

def seed_everything(seed):
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def select_criterion(loss_function):
    if loss_function == 'BCE':
        criterion = # fiil this in # Binary Cross Entropy
    elif loss_function == 'LSGAN':
        criterion = # fiil this in # Least Squares Loss(Means Square Error)
    else:
        raise ValueError('Invalid loss function')

    return criterion

def training(config):
    # TODO: Set the device
    device = # fiil this in # use GPU your own
    print("We are using", device)

    transform = transforms.Compose([
        transforms.Resize(config['model']['image_size']),
        transforms.CenterCrop(config['model']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # TODO: Create the dataloader
    train_data = # fiil this in
    train_loader = # fiil this in

    # TODO: Create the generator, refer './configs/configs.yaml' for the hyperparameters
    netG = Generator(# fiil this in,
                     # fiil this in,
                     # fiil this in,
                     # fiil this in
                     ).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (config['training']['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(config['training']['ngpu'])))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    print(netG)

    # TODO Create the Discriminator, refer './configs/configs.yaml' for the hyperparameters
    netD = Discriminator(# fiil this in,
                         # fiil this in,
                         # fiil this in
                         ).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (config['training']['ngpu'] > 1):
        netD = nn.DataParallel(netD, list(range(config['training']['ngpu'])))

    # Apply the weights_init function to randomly initialize all weights
    # like this: to mean=0, stdev=0.2.
    netD.apply(weights_init)
    print(netD)

    # TODO: Initialize Loss function, refer './configs/configs.yaml' for the hyperparameters
    criterion = select_criterion(# fiil this in).to(device)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, config['model']['nz'], 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 0.9 if config['training']['label_smoothing'] else 1.0
    fake_label = 0.1 if config['training']['label_smoothing'] else 0.0

    # TODO: Setup each optimizers for both G and D, refer './configs/configs.yaml' for the hyperparameters
    optimizerD = # fiil this in
    optimizerG = # fiil this in

    # Training Loop
    print("Starting Training Loop...")
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    # Create an EarlyStopping object
    early_stopping = EarlyStopping(patience=config['training']['patience'], 
                                   verbose=True, 
                                   path=config['paths']['model_save_path'])

    for epoch in range(config['training']['num_epochs']):
        for i, data in enumerate(train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Forward pass real batch through D
            output_real = netD(real_cpu).view(-1)

            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['model']['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            fake = fake.detach()  # Explicitly detach the fake images
            # Classify all fake batch with D
            output_fake = netD(fake).view(-1)

            # Calculate loss and perform backpropagation
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()

            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            errD = errD_real + errD_fake

            D_x = output_real.mean().item()
            D_G_z1 = output_fake.mean().item()
            
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config['model']['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output = netD(fake).view(-1)

            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            errG = criterion(output, label_real)
            errG.backward()

            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, config['training']['num_epochs'], i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Add losses to tensorboard
            writer.add_scalar('Generator_loss', errG.item(), global_step=iters)
            writer.add_scalar('Discriminator_loss', errD.item(), global_step=iters)
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == config['training']['num_epochs']-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Add images to tensorboard
            writer.add_image('Generated Images', img_list[-1].unsqueeze(0), global_step=iters)
            
            iters += 1

            # TODO: Gradient clipping
            # Gradient clipping
            if config['training']['gradient_clipping']:
                # # fiil this in(# fiil this in, 1.0) # Clip the gradient function
                # # fiil this in(# fiil this in, 2.0)
        
        # Create an EarlyStopping object

        early_stopping(errG.item(), netG)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Close the SummaryWriter at the end of your script
    writer.close()

if __name__ == "__main__":
    config = load_config('configs/configs.yaml')  # specify the path to your config file
    seed_everything(config['training']['seed'])
    training(config)
