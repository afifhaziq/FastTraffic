# coding: UTF-8
# Acknowledgements: 
# This project is inspired by the FastText implementation from the repository: [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch).

from cgitb import text
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_fasttraffic import get_time_dif
import wandb


def init_network(model, method='xavier', exclude='embedding', seed=123):

    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, data):

    print(config.train_path.split("\\")[-2])
    wandb.init(project=config.model_name+"-"+config.train_path.split("\\")[-3])
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.num_epochs,
    "batch_size": config.batch_size
    }

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  
    dev_best_loss = float('inf')
    last_improve = 0  
    flag = False 

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step()
        for i, (trains, labels) in enumerate(train_iter):
            #s = time.time()
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            #e = time.time()
            #print(e-s)
            if total_batch % 200 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                wandb.log({"train_loss":  loss.item()})
                wandb.log({"train_acc":  train_acc})
                
                model.train()
                wandb.watch(model)
                
                #time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                model.train()
              

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    end_time = time.time()
    time_dif = (end_time - start_time)/60
    
    average_time = time_dif / config.num_epochs
    print("Training time usage (Minutes):", time_dif)
    wandb.log({"train_time":  float(time_dif)})
    print("Average Traning time (epoch):", average_time)
    wandb.log({"avgtrain_time":  float(average_time)})

    test(config, model, test_iter, data)
    


def test(config, model, test_iter, data):
    # test
    wandb.init(project=config.model_name+"-"+config.train_path.split("\\")[-3]+"-test")
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.num_epochs,
    "batch_size": config.batch_size
    }

    
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    #start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True, data=data)
    #time_dif = get_time_dif(start_time)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and 1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

    # time_dif, average_time = get_time_dif(start_time, test=1, data=data)

    # print(f"Time usage: {time_dif:.10f} seconds")  # Show 6 decimal places
    # print(f"Average time usage: {average_time:.10f} seconds")
    # wandb.log({"test_time":  float(time_dif)})
    # wandb.log({"average_time":  float(average_time)})
    #print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False, data=None):

    model.eval()
    
    start_time = time.time()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    #prob = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
     
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_ = torch.softmax(outputs,dim=1)
            predict_ = predict_.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
            

    acc = metrics.accuracy_score(labels_all, predict_all)
    #time_dif = get_time_dif(start_time)
    if test == True:
        time_dif, average_time = get_time_dif(start_time, test=1, data=data)
        print(f"Testing Time usage: {time_dif:.10f} seconds")  
        print(f"Average Testing time: {average_time:.10f} seconds")
        wandb.log({"test_time":  float(time_dif)})
        wandb.log({"average_time":  float(average_time)})
        
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        F1 = metrics.f1_score(labels_all, predict_all,average='macro')
        f = open(config.save_res,'a')
        f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  \n")
        f.write(report)
        return acc, loss_total / len(data_iter), report, confusion #, F1
        
    return acc, loss_total / len(data_iter)

