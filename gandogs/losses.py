import torch
import torch.nn.functional as F


def GDPPLoss(phiFake, phiReal, backward=True):
    r"""
    Implementation of the GDPP loss. Can be used with any kind of GAN
    architecture.

    Args:

        phiFake (tensor) : last feature layer of the discriminator on real data
        phiReal (tensor) : last feature layer of the discriminator on fake data
        backward (bool)  : should we perform the backward operation ?

    Returns:

        Loss's value. The backward operation in performed within this operator
    """
    def compute_diversity(phi):
        phi = F.normalize(phi, p=2, dim=1)
        SB = torch.mm(phi, phi.t())
        eigVals, eigVecs = torch.symeig(SB, eigenvectors=True)
        return eigVals, eigVecs

    def normalize_min_max(eigVals):
        minV, maxV = torch.min(eigVals), torch.max(eigVals)
        return (eigVals - minV) / (maxV - minV)

    phiFake=phiFake.view(phiFake.size(0),-1)
    phiReal=phiReal.view(phiReal.size(0),-1)
    fakeEigVals, fakeEigVecs = compute_diversity(phiFake)
    realEigVals, realEigVecs = compute_diversity(phiReal)

    # Scaling factor to make the two losses operating in comparable ranges.
    magnitudeLoss = 0.0001 * F.mse_loss(target=realEigVals, input=fakeEigVals)
    structureLoss = -torch.sum(torch.mul(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = normalize_min_max(realEigVals)
    weightedStructureLoss = torch.sum(
        torch.mul(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss

    if backward:
        gdppLoss.backward(retain_graph=True)

    return gdppLoss


# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake, dis_real):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake

def loss_hinge_dis2(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(0.5 - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, dis_real):
  loss = -torch.mean(dis_fake)
  return loss

def loss_rals_dis(dis_fake, dis_real):
  real_label = 0.5
  batch_size=dis_fake.size(0)
  labels = torch.full((batch_size, 1), real_label).cuda()
  loss_real = torch.mean((dis_real - torch.mean(dis_fake) - labels) ** 2)
  loss_fake=torch.mean((dis_fake - torch.mean(dis_real) + labels) ** 2)
  return loss_real, loss_fake

def loss_rals_gen(dis_fake, dis_real):
  real_label = 0.5
  batch_size = dis_fake.size(0)
  labels = torch.full((batch_size, 1), real_label).cuda()
  errG = (torch.mean((dis_real - torch.mean(dis_fake) + labels) ** 2) +
          torch.mean((dis_fake - torch.mean(dis_real) - labels) ** 2)) / 2
  return errG

def loss_hinge_rals_dis(dis_fake, dis_real):
  loss_real1,loss_fake1 =loss_hinge_dis(dis_fake, dis_real)
  loss_real2,loss_fake2 =loss_rals_dis(dis_fake, dis_real)
  loss_real=loss_real1+loss_real2
  loss_fake=loss_fake1+loss_fake2
  return loss_real/2, loss_fake/2

def loss_hinge_rals_gen(dis_fake, dis_real):
  loss1 = loss_hinge_gen(dis_fake, dis_real)
  loss2 = loss_rals_gen(dis_fake, dis_real)
  return (loss1+loss2)/2

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis