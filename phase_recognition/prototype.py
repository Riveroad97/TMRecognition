
import os
import time
from tracemalloc import start
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from thop import profile
from sklearn.metrics import f1_score

from utils import fusion, segment_bars_with_confidence_score


loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')

fn_tonumpy = lambda x: x.detach().cpu().numpy()


def hierarch_train(args, model, train_loader, validation_loader, device, save_dir = 'models', debug = False):
   
    model.to(device)
    num_classes = args.num_classes


    save_dir = os.path.join(save_dir, args.model)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_epoch = 0
    best_acc = 0
    best_f1 = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        if epoch % 30 == 0:
            args.learning_rate = args.learning_rate * 0.5
       
        
        correct = 0
        total = 0
        loss_item = 0
        ce_item = 0 
        ms_item = 0
        lc_item = 0
        gl_item = 0
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
        max_seq = 0
        mean_len = 0
        ans = 0
        max_phase = 0

        for (video, labels, video_name) in (train_loader):
            labels = torch.Tensor(labels).long()
            
                
            video, labels = video.to(device), labels.to(device)


 
            predicted_list, feature_list, prototype = model(video)
            
            mean_len += predicted_list[0].size(-1)
            ans += 1
            all_out, resize_list, labels_list = fusion(predicted_list,labels, args)

            max_seq = max(max_seq, video.size(1))
            
            loss = 0 
            
            if args.ms_loss:
                ms_loss = 0
                
                for p,l in zip(resize_list,labels_list):
                    ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                    ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                loss = loss + ms_loss
                ms_item += ms_loss.item()

            optimizer.zero_grad()
            loss_item += loss.item()

            
            if args.last:
                all_out =  resize_list[-1]
            if args.first:
                all_out = resize_list[0]
        
            loss.backward()

            optimizer.step()
            
            _, predicted = torch.max(all_out.data, 1)
            
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]


        print('Train Epoch {}: Acc {}, Loss {}, ms {}'.format(epoch, correct / total, loss_item /total,  ms_item/total))
        if debug:
            # save_dir
            test_acc, test_f1, predicted, out_pro, test_video_name=hierarch_test(args, model, validation_loader, device)
            # if test_acc > best_acc:
                # best_acc = test_acc
                # best_epoch = epoch
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/best.model')
        print('Best Test: F1 {} Acc {},  Epoch {}'.format(best_f1, best_acc, best_epoch))


def hierarch_test(args, model, test_loader, device, random_mask=False):
   
    model.to(device)
    num_classes = args.num_classes
    
    model.eval()
   
    with torch.no_grad():
        correct = 0
        total = 0
        loss_item = 0
        
        center = torch.ones((1, 64, num_classes), requires_grad=False)
        center = center.to(device)
        label_correct={}
        label_total= {}
        probabilty_list = []
        video_name_list=[]
        precision=0
        recall = 0
        ce_item = 0 
        ms_item = 0
        lc_item = 0
        gl_item = 0
        max_seq = 0

        all_preds = []
        all_labels = []

        for n_iter,(video, labels, video_name ) in enumerate(test_loader):
            
                
            labels = torch.Tensor(labels).long()
            
                
            video, labels = video.to(device), labels.to(device)
            max_seq = max(max_seq, video.size(1))

            predicted_list, feature_list, _ = model(video)
            
            all_out, resize_list,labels_list = fusion(predicted_list,labels, args)
            
            loss = 0 

            
            
            if args.ms_loss:
                ms_loss = 0
                for p,l in zip(resize_list,labels_list):
                    # print(p.size())
                    ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, args.num_classes), l.view(-1))
                    ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                loss = loss + ms_loss
                ms_item += ms_loss.item()
            
            
            loss_item += loss.item()

            if args.last:
                all_out =  resize_list[-1]
            if args.first:
                all_out = resize_list[0]

            _, predicted = torch.max(all_out.data, 1)
            

            predicted = predicted.squeeze()

            # labels = labels_list[-1]
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            
            video_name_list.append(video_name)


            

            all_out = F.softmax(all_out,dim=1)

            probabilty_list.append(all_out.transpose(1,2))


            all_preds += predicted.squeeze(0).tolist()
            all_labels += [label.item() for label in labels]
        
        f1   = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')


        print('Test  Acc {}, F1 {}, Loss {}, ms {}'.format( correct / total, f1, loss_item /total, ms_item/total))
        # print('BMG precision {}, BMG recall {}'.format(precision/(n_iter+1), recall/(n_iter+1) ))
        # print(len(label_total))
        for (kc, vc), (kall, vall) in zip(label_correct.items(),label_total.items()):
            print("{} acc: {}".format(kc, vc/vall))
        return correct / total, f1, all_preds, probabilty_list, video_name_list


def hierarch_predict(model, args, device,test_loader, name, pki = False, split='test'):
    
    phase2label_dicts = {
    'ent6':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4,
    'no_drill':5,
    'background':6
    },
    'ent5':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4
    },
    
}
    model.to(device)
    model.eval()

    
    pic_save_dir = 'experiments/results/{}/{}/train/vis/'.format(name, args.model)
    results_dir = 'experiments/results/{}/{}/train/prediction/'.format(name, args.model)

    gt_dir = args.dataset_path+'/Label/frame_label'

    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()
            print(video.size(),video_name,labels.size())

            video = video.to(device)
            labels = labels.to(device)
            predicted_list, feature_list, _ = model(video)

            all_out, resize_list, labels_list = fusion(predicted_list, labels, args)
            if args.last:
                all_out =  resize_list[-1]
            if args.first:
                all_out = resize_list[0]
            confidence, predicted = torch.max(F.softmax(all_out.data,1), 1)

            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            predicted = predicted.squeeze(0).tolist()
            
            confidence = confidence.squeeze(0).tolist()
            
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])

            predicted_phases_expand = []

            for i in predicted:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i]*5 )) # we downsample the framerate from 30fps to 5fps

            print(video_name)
         
            v_n = video_name[0].split('.')[0]
            # v_n = re.findall(r"\d+\.?\d*",v_n)

            # v_n = float(v_n[0])
            target_video_file = f"{v_n}_pred.txt"
            print(target_video_file)

            gt_file = f"{v_n}.txt"

            g_ptr = open(os.path.join(gt_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')


            for index, line in enumerate(predicted):
                phase_dict = phase2label_dicts[args.dataset]
                p_phase = ''
                for k, v in phase_dict.items():
                    if v == int(line):
                        p_phase = k
                        break

                f_ptr.write('{}\t{}\n'.format(index, p_phase))
            f_ptr.close()


def base_train(args, model, train_loader, validation_loader, device, save_dir = 'models', debug = False):

    model.to(device)

    save_dir = os.path.join(save_dir, args.model)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_epoch = 0
    best_acc = 0
    best_f1 = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        if epoch % 30 == 0:
            args.learning_rate = args.learning_rate * 0.5
        correct = 0
        total = 0
        loss_item = 0
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
        for (video, labels, video_name) in (train_loader):
            labels = torch.Tensor(labels).long()

            video, labels = video.to(device), labels.to(device)
            outputs = model(video)
            
            loss = 0
            loss += loss_layer(outputs.transpose(2, 1).contiguous().view(-1, args.num_classes), labels.view(-1)) # cross_entropy loss
            loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(outputs[:, :, 1:], dim=1), F.log_softmax(outputs.detach()[:, :, :-1], dim=1)), min=0, max=16)) # smooth loss

            loss_item += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item / total))
        if debug:
            test_acc, test_f1 = base_test(model, validation_loader, device)
            if test_f1 > best_f1:
                best_f1 = test_f1
            # if test_acc > best_acc:
                # best_acc = test_acc
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/best.model')
        # print('Best Test: F1 {}, Epoch {}'.format(best_f1, best_epoch))
        print('Best Test: F1 {} Acc {},  Epoch {}'.format(best_f1, best_acc, best_epoch))


def base_test(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for (video, labels, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            video, labels = video.to(device), labels.to(device)
            outputs = model(video)
            _, predicted = torch.max(outputs.data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            all_preds += predicted.squeeze(0).tolist()
            all_labels += [label.item() for label in labels]

        f1   = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
        
        print('Test  Acc {}, F1 {}'.format( correct / total, f1))

        # print('Test: Acc {}'.format(correct / total))
        return correct / total, f1


def base_predict(model, args, device, test_loader, name):
    phase2label_dicts = {
    'ent6':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4,
    'no_drill':5,
    'background':6
    },
    'ent5':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4
    },
    
}
    model.to(device)
    model.eval()

    
    pic_save_dir = 'experiments/results/{}/{}/train/vis/'.format(name, args.model)
    results_dir = 'experiments/results/{}/{}/train/prediction/'.format(name, args.model)

    gt_dir = args.dataset_path+'/Label/frame_label'

    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with torch.no_grad():
        correct = 0
        total = 0
        total_label = []
        total_pred = []
        for (video, labels, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            video, labels = video.to(device), labels.to(device)
            outputs = model(video)
            confidence, predicted = torch.max(F.softmax(outputs.data,1), 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()

            labels = [label.item() for label in labels]

            total_label += labels
            total_pred += predicted

            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])

            print(video_name)

            v_n = video_name[0].split('.')[0]
            target_video_file = f"{v_n}_pred.txt"
            print(target_video_file)

            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')

            for index, line in enumerate(predicted):
                phase_dict = phase2label_dicts[args.dataset]
                p_phase = ''
                for k, v in phase_dict.items():
                    if v == int(line):
                        p_phase = k
                        break

                f_ptr.write('{}\t{}\n'.format(index, p_phase))
            f_ptr.close()


def refine_train(args, base_model, refine_model, train_loader, validation_loader, device, save_dir = 'models', debug = False):
   
    base_model.to(device)
    refine_model.to(device)

    num_classes = args.num_classes

    save_dir = os.path.join(save_dir, args.refine_model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.refine_model == 'gru':
        r_lr = args.refine_learning_rate *10
    elif args.refine_model == 'causal_tcn':
        r_lr = args.refine_learning_rate
    elif args.refine_model == 'tcn':
        r_lr = args.refine_learning_rate
    
    best_epoch = 0
    best_f1 = 0
    
    base_model.eval()
    for epoch in range(1, args.refine_epochs + 1):
        total = 0
        correct = 0
        loss_item = 0

        optimizer = torch.optim.Adam(refine_model.parameters(), r_lr, weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for (video, labels, video_name) in (train_loader):
            labels = torch.Tensor(labels).long()
            video, labels = video.to(device), labels.to(device)

            with torch.no_grad():
                outputs1 = base_model(video)
            
            refine_model.train()
            outputs2, _ = refine_model(outputs1)
            
            loss = 0
            for output in outputs2:
                loss += loss_layer(output.transpose(2,1).contiguous().view(-1, num_classes), labels.view(-1))
            
            loss_item += loss.item()
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs2[-1].data, 1)
            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
            
        scheduler.step(loss_item)

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct / total, loss_item /total))
        if debug:
            # save_dir
            test_f1 = refine_test(args, base_model, refine_model, validation_loader, device)
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(refine_model.state_dict(), save_dir + '/best.model'.format(epoch))
        print('Best Test: F1 {}, Epoch {}'.format(best_f1, best_epoch))


def refine_test(args, base_model, refine_model, test_loader, device):
    # global device
    base_model.to(device)
    refine_model.to(device)

    base_model.eval()
    refine_model.eval()
    with torch.no_grad():
        correct1 = 0
        correct2 = 0
        total = 0
        pred_list1 = []
        pred_list2 = []
        all_labels = []
        for (video, labels, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            video, labels = video.to(device), labels.to(device)

            outputs1 = base_model(video)
            outputs2, _ = refine_model(outputs1)

            
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2[-1].data, 1)
            correct1 += ((predicted1 == labels).sum()).item()
            correct2 += ((predicted2== labels).sum()).item()
            total += labels.shape[0]
            

            pred_list1 += predicted1.squeeze(0).tolist()
            pred_list2 += predicted2.squeeze(0).tolist()
            all_labels += [label.item() for label in labels]
        
        f1_pred1   = f1_score(y_true=all_labels, y_pred=pred_list1, average='macro')
        f1_pred2   = f1_score(y_true=all_labels, y_pred=pred_list2, average='macro')

        print('Test: Base model Acc {} F1 {}, Prior Model Acc {} F1 {}'.format(correct1 / total, f1_pred1, correct2/total, f1_pred2))
        return f1_pred2


def refine_predict(base_model, refine_model, args, device, test_loader, name, pki = False, split='test'):
    
    phase2label_dicts = {
    'ent6':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4,
    'no_drill':5,
    'background':6
    },
    'ent5':{
    'Cortical_drilling':0,
    'Drilling_of_the_mastoid':1,
    'Drilling_the_antrum':2,
    'Drilling_near_the_facial_nerve':3,
    'Posterior_tympanotomy':4
    },
    
}
    base_model.eval()
    refine_model.eval()
    base_model.to(device)
    refine_model.to(device)


    pic_save_dir = 'experiments/results/{}/{}/refine/vis/'.format(name, args.refine_model)
    results_dir = 'experiments/results/{}/{}/refine/prediction/'.format(name, args.refine_model)

    gt_dir = args.dataset_path+'/frame_label'

    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with torch.no_grad():
        for (video, labels, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()
            video = video.to(device)
            labels = labels.to(device)
 
            base_model_output = base_model(video)

            confidence, base_predicted = torch.max(F.softmax(base_model_output.data,1), 1)
            base_predicted = base_predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()

            outputs, _ = refine_model(base_model_output)
            confidence, predicted = torch.max(F.softmax(outputs[-1].data,1), 1)
            
            predicted = predicted.squeeze(0).tolist()
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)

            confidence = confidence.squeeze(0).tolist()

            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, base_predicted, predicted])

            v_n = video_name[0].split('.')[0]
            target_video_file = f"{v_n}_pred.txt"
            print(target_video_file)

            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')


            for index, line in enumerate(predicted):
                phase_dict = phase2label_dicts[args.dataset]
                p_phase = ''
                for k, v in phase_dict.items():
                    if v == int(line):
                        p_phase = k
                        break

                f_ptr.write('{}\t{}\n'.format(index, p_phase))
            f_ptr.close()










