import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from check_dataset import check_dataset
from check_model import check_model
from utils.utils import AverageMeter, accuracy, set_logging_config
from train.meta_optimizers import MetaSGD
from models.basenet import Predictor, Predictor_deep

torch.backends.cudnn.benchmark = True


def _get_num_features(model):
    if model.startswith('resnet'):
        n = int(model[6:])
        if n in [18, 34, 50, 101, 152]:
            return [64, 64, 128, 256, 512]
        else:
            n = (n-2) // 6
            return [16]*n+[32]*n+[64]*n
    elif model.startswith('vgg'):
        n = int(model[3:].split('_')[0])
        if n == 9:
            return [64, 128, 256, 512, 512]
        elif n == 11:
            return [64, 128, 256, 512, 512]

    raise NotImplementedError


class FeatureMatching(nn.ModuleList):
    def __init__(self, source_model, target_model, pairs):
        super(FeatureMatching, self).__init__()
        self.src_list = _get_num_features(source_model)
        self.tgt_list = _get_num_features(target_model)
        self.pairs = pairs

        for src_idx, tgt_idx in pairs:
            self.append(nn.Conv2d(self.tgt_list[tgt_idx], self.src_list[src_idx], 1))

    def forward(self, source_features, target_features,
                weight, beta, loss_weight):

        matching_loss = 0.0
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            else:
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2)
            if loss_weight is None and weight is None:
                diff = diff.mean(1).mean(0).mul(beta[i])
            elif loss_weight is None:
                diff = diff.mul(weight[i]).sum(1).mean(0).mul(beta[i])
            elif weight is None:
                diff = (diff.sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            else:
                diff = (diff.mul(weight[i]).sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            matching_loss = matching_loss + diff
        return matching_loss


class WeightNetwork(nn.ModuleList):
    def __init__(self, source_model, pairs):
        super(WeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        for i, _ in pairs:
            self.append(nn.Linear(n[i], n[i]))
            self[-1].weight.data.zero_()
            self[-1].bias.data.zero_()
        self.pairs = pairs

    def forward(self, source_features):
        outputs = []
        for i, (idx, _) in enumerate(self.pairs):
            f = source_features[idx]
            f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
            outputs.append(F.softmax(self[i](f), 1))
        return outputs


class LossWeightNetwork(nn.ModuleList):
    def __init__(self, source_model, pairs, weight_type='relu', init=None):
        super(LossWeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        if weight_type == 'const':
            self.weights = nn.Parameter(torch.zeros(len(pairs)))
        else:
            for i, _ in pairs:
                l = nn.Linear(n[i], 1)
                if init is not None:
                    nn.init.constant_(l.bias, init)
                self.append(l)
        self.pairs = pairs
        self.weight_type = weight_type

    def forward(self, source_features):
        outputs = []
        if self.weight_type == 'const':
            for w in F.softplus(self.weights.mul(10)):
                outputs.append(w.view(1, 1))
        else:
            for i, (idx, _) in enumerate(self.pairs):
                f = source_features[idx]
                f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
                if self.weight_type == 'relu':
                    outputs.append(F.relu(self[i](f)))
                elif self.weight_type == 'relu-avg':
                    outputs.append(F.relu(self[i](f.div(f.size(1)))))
                elif self.weight_type == 'relu6':
                    outputs.append(F.relu6(self[i](f)))
        return outputs


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def adentropy(F1,feat,lamda,eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def str2bool(v):
    return v.lower() in ('true')

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataroot', required=True, help='Path to the dataset')
    parser.add_argument('--dataset', default='cub200')
    parser.add_argument('--datasplit', default='cub200')
    parser.add_argument('--batchSize', type=int, default=64, help='Input batch size')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--source-model', default='resnet34', type=str)
    parser.add_argument('--source-domain', default='imagenet', type=str)
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default='resnet18', type=str)
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--wnet-path', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1,help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--schedule', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pairs', type=str, default='4-4,4-3,4-2,4-1,3-4,3-3,3-2,3-1,2-4,2-3,2-2,2-1,1-4,1-3,1-2,1-1')

    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Initial learning rate for meta networks')
    parser.add_argument('--meta-wd', type=float, default=1e-4)
    parser.add_argument('--loss-weight', action='store_true', default=True)
    parser.add_argument('--loss-weight-type', type=str, default='relu6')
    parser.add_argument('--loss-weight-init', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--experiment', default='logs', help='Where to store models')

    ## DA

    parser.add_argument('--source', type=str, default='real')
    parser.add_argument('--target', type=str, default='sketch')
    parser.add_argument('--method', type=str, default='MME')
    parser.add_argument('--pretrained', type=str2bool, default=False)


    # default settings
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.experiment)
    set_logging_config(opt.experiment)
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(opt)

    # load source model
    if opt.source_domain == 'imagenet':
        from models import resnet_ilsvrc
        source_model = resnet_ilsvrc.__dict__[opt.source_model](pretrained=True).to(device)
    else:
        opt.model = opt.source_model
        weights = []
        source_gen_params = []

        source_path = '/home/pjw/projects/what_and_where_to_transfer/source_models/G_iter_model_resnet34_MME_real_to_sketch_step_40000.pth.tar'
        # source_path = os.path.join(
        #     opt.source_path, '{}-{}'.format(opt.source_domain, opt.source_model),
        #     '0',
        #     'model_best.pth.tar'
        # )
        ckpt = torch.load(source_path)
        #print(ckpt.keys())
        opt.num_classes = 345  #ckpt['num_classes']
        from models import resnet_ilsvrc
        source_model = resnet_ilsvrc.__dict__[opt.source_model](pretrained=True).to(device) # check_model(opt).to(device)
        source_model.load_state_dict(ckpt, strict=False)

    pairs = []
    for pair in opt.pairs.split(','):
        pairs.append((int(pair.split('-')[0]),
                      int(pair.split('-')[1])))

    wnet = WeightNetwork(opt.source_model, pairs).to(device)
    weight_params = list(wnet.parameters())
    if opt.loss_weight:
        lwnet = LossWeightNetwork(opt.source_model, pairs, opt.loss_weight_type, opt.loss_weight_init).to(device)
        weight_params = weight_params + list(lwnet.parameters())

    if opt.wnet_path is not None:
        ckpt = torch.load(opt.wnet_path)
        wnet.load_state_dict(ckpt['w'])
        if opt.loss_weight:
            lwnet.load_state_dict(ckpt['lw'])

    if opt.optimizer == 'sgd':
        source_optimizer = optim.SGD(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd, momentum=opt.momentum, nesterov=opt.nesterov)
    else:
        source_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)

    # load dataloaders
    loaders = check_dataset(opt)

    # load target model
    opt.model = opt.target_model
    # target_model = check_model(opt).to(device)
    # target G, target F
    target_G = check_model(opt).to(device)
    target_F = Predictor(num_class=opt.num_classes, inc=512, temp=1).to(device)

    weights_init(target_F)

    target_branch = FeatureMatching(opt.source_model,
                                    opt.target_model,
                                    pairs).to(device)

    target_params = list(target_G.parameters()) + list(target_F.parameters()) + list(target_branch.parameters())
    target_G_params = list(target_G.parameters())
    target_F_params = list(target_F.parameters())

    if opt.meta_lr == 0:
        target_G_optimizer = optim.SGD(target_params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
    else:
        target_optimizer = MetaSGD(target_params,
                                   [target_G, target_branch],
                                   lr=opt.lr,
                                   momentum=opt.momentum,
                                   weight_decay=opt.wd, rollback=True, cpu=opt.T>2)

        target_G_optimizer = optim.SGD(target_G_params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
        target_F_optimizer = optim.SGD(target_F_params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)

    state = {
        'target_G': target_G.state_dict(),
        'target_F': target_F.state_dict(),
        'target_branch': target_branch.state_dict(),
        'target_optimizer': target_G_optimizer.state_dict(),
        'w': wnet.state_dict(),
        'best': (0.0, 0.0, 0.0)
    }
    if opt.loss_weight:
        state['lw'] = lwnet.state_dict()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(target_optimizer, opt.epochs)
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(target_G_optimizer, opt.epochs)
    F_scheduler = optim.lr_scheduler.CosineAnnealingLR(target_F_optimizer, opt.epochs)

    def validate(G, F, loader):
        acc = AverageMeter()
        G.eval()
        F.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            features, _ = G.forward_features(x)
            y_pred = F(features)
            acc.update(accuracy(y_pred.data, y, topk=(1,))[0].item(), x.size(0))
        return acc.avg

    def inner_objective(data, matching_only=False):
        x, y = data[0].to(device), data[1].to(device)
        # MME
        features, target_features = target_G.forward_features(x)
        y_pred = target_F(features)

        with torch.no_grad():
            s_pred, source_features = source_model.forward_with_features(x)

        weights = wnet(source_features)
        state['loss_weights'] = ''
        if opt.loss_weight:
            loss_weights = lwnet(source_features)
            state['loss_weights'] = ' '.join(['{:.2f}'.format(lw.mean().item()) for lw in loss_weights])
        else:
            loss_weights = None
        beta = [opt.beta] * len(wnet)

        matching_loss = target_branch(source_features,
                                      target_features,
                                      weights, beta, loss_weights)

        state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()

        if matching_only:
            return matching_loss

        # cross entropy
        loss = F.cross_entropy(y_pred, y)
        state['loss'] = loss.item()

        return loss + matching_loss

    def outer_objective(data):
        x, y = data[0].to(device), data[1].to(device)
        ### MME
        features, _ = target_G.forward_features(x)
        y_pred = target_F(features)

        state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()

        # cross entropy
        loss = F.cross_entropy(y_pred, y)
        #
        state['loss'] = loss.item()
        return loss

    # source generator training
    state['iter'] = 0
    for epoch in range(opt.epochs):
        if opt.schedule:
            scheduler.step()
            G_scheduler.step()
            F_scheduler.step()

        state['epoch'] = epoch
        target_G.train()
        target_F.train()
        source_model.eval()

        source_loader = loaders[0]
        target_loader = loaders[1]
        unl_target_loader = loaders[2]

        data_iter_s = iter(source_loader)
        data_iter_t = iter(target_loader)
        data_iter_t_unl = iter(target_loader)

        len_train_source = len(source_loader)
        len_train_target = len(target_loader)
        len_train_target_semi = len(unl_target_loader)

        steps = 1000
        for step in range(steps):

            if step % len_train_target == 0:
                data_iter_t = iter(target_loader)
            if step % len_train_target_semi == 0:
                data_iter_t_unl = iter(unl_target_loader)
            if step % len_train_source == 0:
                data_iter_s = iter(source_loader)

            data_t = next(data_iter_t)
            data_t_unl = next(data_iter_t_unl)
            data_s = next(data_iter_s)

            img_con = torch.cat((data_t[0], data_s[0]), 0)
            label_con = torch.cat((data_t[1], data_s[1]), 0)
            data = img_con, label_con

            target_optimizer.zero_grad()
            inner_objective(data).backward() # cre + feature matching loss
            target_optimizer.step(None)

            ## MME for F1
            if not opt.method == 'S+T':

                tu_unl_features, _ = target_G.forward_features(data_t_unl[0].to(device))
                if opt.method == 'MME':
                    loss_t = adentropy(target_F, tu_unl_features, 0.1)
                    loss_t.backward()
                    target_G_optimizer.step()
                    target_F_optimizer.step()

            logger.info('[Epoch {:3d}] [Iter {:3d}] [Loss {:.4f}] [Acc {:.4f}] [LW {}]'.format(
                state['epoch'], state['iter'],
                state['loss'], state['accuracy'], state['loss_weights']))
            state['iter'] += 1

            for _ in range(opt.T):
                target_optimizer.zero_grad()
                target_optimizer.step(inner_objective, data, True)  # feature matching loss만 학습

            target_optimizer.zero_grad()
            target_optimizer.step(outer_objective, data) # update theta, minimize cre

            target_optimizer.zero_grad()
            source_optimizer.zero_grad()
            outer_objective(data).backward() # update w,lamda, minimize cre
            target_optimizer.meta_backward()
            source_optimizer.step()

        acc = (validate(target_G, target_F, loaders[0]),
               validate(target_G, target_F, loaders[1]),
               validate(target_G, target_F, loaders[3]))

        if state['best'][2] < acc[2]:
            state['best'] = acc

        if state['epoch'] % 10 == 0:
            torch.save(state, os.path.join(opt.experiment, 'ckpt-{}.pth'.format(state['epoch']+1)))

        logger.info('[Epoch {}] [src_val {:.4f}] [trg_val {:.4f}] [test {:.4f}] [best {:.4f}]'
                    .format(epoch, acc[0], acc[1], acc[2], state['best'][2]))


if __name__ == '__main__':
    main()
