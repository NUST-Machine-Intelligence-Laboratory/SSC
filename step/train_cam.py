
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import importlib

import voc12.dataloader_new
from misc import pyutils, torchutils, loss_util

import numpy as np
import cv2

def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x,_ = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss1')))

    return

def mx_norm(pred):
    n, c, _= pred.shape
    pred = torch.clamp(pred,0)
    pred = pred.reshape(n, c, -1)
    pred_max = torch.max(pred, dim=2, keepdim=True)[0]
    pred /= pred_max + 1e-5
    return pred


def alignment_loss(pred, sp, label, criterion):
    
    superpixel = sp.float()
    
    superpixel = F.interpolate(superpixel.unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1)
    _, h2, w2 = superpixel.shape

    n, c, _, _ = pred.shape

    sp_num = int(superpixel.max().numpy()) + 1
    superpixel = superpixel.to(torch.int64)
    sp_flat = F.one_hot(superpixel, num_classes=sp_num)
    sp_flat = sp_flat.permute(0, 3, 1, 2)
    pred = F.interpolate(pred, size=(h2, w2), mode='bilinear')
 
    sp_flat = sp_flat.reshape(n, sp_num, -1)   
    score_flat = pred.reshape(n, c, -1) * label.reshape(n, c, 1)  
    sp_flat_avg = sp_flat / np.maximum(sp_flat.sum(axis=2, keepdims=True), 1e-5)  
    sp_flat_avg = sp_flat_avg.float()
    score_sum = torch.matmul(score_flat, torch.transpose(sp_flat, 1, 2).cuda().float())     
    score_sum = score_sum.reshape(n, c, sp_num, 1)         
    sp_flat_avg = sp_flat_avg.unsqueeze(1).repeat(1, c, 1, 1) 
    final_averaged_value = torch.mul(sp_flat_avg.cuda(), score_sum)    
    final_averaged_value = final_averaged_value.sum(2) 
    final_averaged_value = final_averaged_value * label.reshape(n, c, 1)  
    original_value = mx_norm(score_flat)  
    final_averaged_value = mx_norm(final_averaged_value)

    original_value = original_value[label==1]
    oriv = original_value.detach().cpu().numpy()
    oriv = oriv.reshape(-1, h2, w2)
    thr = 0.3
    for i in range(oriv.shape[0]):
        single_att = oriv[i]
        region = np.zeros_like(single_att)
        region[single_att > thr] = 1.0
        kernel = np.ones((16, 16), np.uint8)
        region = cv2.erode(region.astype('uint8'), kernel, iterations=1)
        oriv[i] = oriv[i] * region
    
    oriv = torch.from_numpy(oriv).reshape(-1, h2*w2).cuda()

        
    final_averaged_value = final_averaged_value[label==1].detach()
    final_averaged_value = torch.where(final_averaged_value > oriv, final_averaged_value, oriv)
    a_loss =  criterion(original_value, final_averaged_value)

    return a_loss.float()


def run(args):

    mse_loss = nn.MSELoss(size_average=True).cuda()

    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model_dec = getattr(importlib.import_module('net.decoder'), 'Decoder')()
    


    train_dataset = voc12.dataloader_new.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader_new.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    model_dec = getattr(importlib.import_module('net.decoder'), 'Decoder')()
    learning_rate_rec = 0.0005
    power             = 0.9
    dec_s_opt = optim.Adam(model_dec.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))

    model_dec = torch.nn.DataParallel(model_dec).cuda()
    model_dec.train()

    rec_opt_list = []
    rec_opt_list.append(dec_s_opt)
    

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    criterion = torch.nn.MSELoss()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            adjust_learning_rate(rec_opt_list , base_lr=learning_rate_rec, i_iter=step+ep*len(train_data_loader), max_iter=args.cam_num_epoches*len(train_data_loader), power=power)

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            label_sp = pack['spixel']

            x,x_int = model(img)
            loss_cls = F.multilabel_soft_margin_loss(x, label)

            x_rec = model_dec(x_int)
            VGG_loss = loss_util.VGGLoss()
            loss_per = VGG_loss(x_rec, img.cuda())

            

            if ep<2:   #warm up
                avg_meter.add({'loss_cls': loss_cls.item()})
                avg_meter.add({'loss_per': loss_per.item()})
                
                loss = loss_cls +  loss_per 

                optimizer.zero_grad()
                dec_s_opt.zero_grad()
                loss.backward()
                optimizer.step()
                dec_s_opt.step()

                

                if (optimizer.global_step-1)%100 == 0:
                    timer.update_progress(optimizer.global_step / max_step)

                    print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                        'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                        'loss_per:%.4f' % (avg_meter.pop('loss_per')),
                        'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                        'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                        'etc:%s' % (timer.str_estimated_complete()), flush=True)
            else:

                loss_align = alignment_loss(x_int, label_sp, label, criterion)


                avg_meter.add({'loss_cls': loss_cls.item()})
                avg_meter.add({'loss_per': loss_per.item()})
                avg_meter.add({'loss_align': loss_align.item()})

                loss = loss_cls +  loss_per +  loss_align

                optimizer.zero_grad()
                dec_s_opt.zero_grad()
                loss.backward()
                optimizer.step()
                dec_s_opt.step()

                

                if (optimizer.global_step-1)%100 == 0:
                    timer.update_progress(optimizer.global_step / max_step)

                    print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                        'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                        'loss_per:%.4f' % (avg_meter.pop('loss_per')),
                        'loss_align:%.4f' % (avg_meter.pop('loss_align')),
                        'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                        'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                        'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            '''validate(model, val_data_loader)'''
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.save(model_dec.module.state_dict(), args.cam_weights_name + '_dec.pth')
    torch.cuda.empty_cache()