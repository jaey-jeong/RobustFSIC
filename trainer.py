import torch
from torch.utils.data import DataLoader
import torchvision
#from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent)

from tqdm import tqdm
import numpy as np
import os

from pgd import projected_gradient_descent
from model.network import resnet12, classification_head, feature_purification_network
from utils.dataloader import FSICdataloader
from utils.utils import adversarial_reweighting, adversarial_reweighting_crossentropy, cross_entropy

def train() :
    print("Start train !")

    train_path = '../miniimagenet/mini-imagenet-cache-train.pkl'
    val_path = '../miniimagenet/mini-imagenet-cache-val.pkl'
    #test_path = '../miniimagenet/mini-imagenet-cache-test.pkl'

    train_data = FSICdataloader(train_path)
    val_data = FSICdataloader(val_path)
    #test_data = FSICdataloader(test_path) # 나중에 변경 해야함
    train_data_size, train_class_num = train_data.__len__(), train_data.class_size()
    val_data_size, val_class_num = val_data.__len__(), val_data.class_size()
    ssl_class_num = 2
    print(train_data_size, train_class_num)
    print(val_data_size, val_class_num)
    #print(test_data.__len__())
    train_loader = DataLoader(train_data, 
                            batch_size = 600, 
                            num_workers = 4,
                            shuffle = True,
                            pin_memory = True)
    
    val_loader = DataLoader(val_data, 
                            batch_size = 600, 
                            num_workers = 4,
                            shuffle = False,
                            pin_memory = True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, clf_head_adv_reweight, clf_head_adv_aware, purifier = get_model(train_class_num, val_class_num, ssl_class_num)
    #print(model)

    print("Set Optimizer .. ")
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_cls = criterion_cls.cuda()
    criterion_pur = torch.nn.MSELoss()
    criterion_pur = criterion_pur.cuda()
    params = list(model.parameters()) + list(clf_head_adv_reweight.parameters()) \
                        + list(clf_head_adv_aware.parameters()) + list(purifier.parameters())
    optimizer = torch.optim.SGD(params, lr = 0.05, weight_decay = 0, momentum=0.9)

    for epoch in tqdm(range(1, 100+1)) : #embedding network training for 100 epochs
        
        model.train()
        clf_head_adv_reweight.train()
        clf_head_adv_aware.train()
        purifier.train()
        #mode = 'train'
        #generation_mode = True
        running_total_loss = 0.0
        running_adv_clf_loss = 0.0
        running_ssl_loss = 0.0
        running_pr_loss = 0.0
        adv_correct_sum = 0
        ssl_correct_sum = 0
        aware_correct_sum = 0
        iter_cnt = 0
        #running_acc = 0.0
        
        for imgs, targets in tqdm(train_loader) :
            #continue
            iter_cnt += 1
            targets = np.asarray(targets)
            targets = torch.from_numpy(targets.astype('long'))
            #print(np.shape(targets))
            targets = targets.to(device)
            imgs = imgs.to(device)
            imgs = imgs.float()
            optimizer.zero_grad()

            ## Adversary generation
            adv_imgs, adv_out, adv_gen_loss= projected_gradient_descent(model, clf_head_adv_reweight, 'train', 
                                            imgs, targets, cross_entropy, 7, 2.0, 'inf', 0.01, 'inf', train_class_num) #image
            #adv_out = adv_out

            ## Feature extraction
            clean_feature = model(imgs)
            adv_feature = model(adv_imgs)

            ## Adversarial-reweighted training
            w = adversarial_reweighting(adv_gen_loss, 7)
            w = torch.from_numpy(w.astype('float'))
            adv_out = clf_head_adv_reweight(adv_feature)
            #print(np.shape(w))
            
            loss_ar = adversarial_reweighting_crossentropy(targets, 
                                                            adv_out.cpu(), 
                                                            w, train_class_num)
            #loss_ar = torch.tensor(loss_ar, requires_grad = True)
            running_adv_clf_loss += loss_ar

            ## Adversarial-aware training
            clean_targets = torch.from_numpy(np.zeros_like(targets.cpu()))
            adv_targets = torch.from_numpy(np.ones_like(targets.cpu()))
            #print(clean_targets, adv_targets)
            clean_targets = clean_targets.to(device)
            adv_targets = adv_targets.to(device)
            ssl_clean_aware_out, clean_aware_out = clf_head_adv_aware(clean_feature) # legitimate 0
            ssl_adv_aware_out, adv_aware_out = clf_head_adv_aware(adv_feature) # adversarial 1
            #print(ssl_clean_aware_out, ssl_adv_aware_out)
            ssl_clean_loss_aa = cross_entropy(clean_targets.cpu(), ssl_clean_aware_out.cpu(), ssl_class_num)
            ssl_adv_loss_aa = cross_entropy(adv_targets.cpu(), ssl_adv_aware_out.cpu(), ssl_class_num)
            clean_loss_aa = cross_entropy(targets.cpu(), clean_aware_out.cpu(), train_class_num)
            adv_loss_aa = cross_entropy(targets.cpu(), adv_aware_out.cpu(), train_class_num)
            loss_aa = clean_loss_aa + adv_loss_aa + ssl_clean_loss_aa + ssl_adv_loss_aa
            running_ssl_loss += loss_aa
 
            ## Feature purification
            purified_clean_feature = purifier(clean_feature)
            purified_adv_feature = purifier(adv_feature)
            
            clean_pur_loss = criterion_pur(purified_clean_feature, clean_feature)
            adv_pur_loss = criterion_pur(purified_adv_feature, clean_feature)
            loss_pr = clean_pur_loss + adv_pur_loss
            running_pr_loss += loss_pr

            ## Overall loss
            loss = loss_ar + 0.5*loss_aa + 0.3*loss_pr
            loss.backward()
            #print(adv_out)

            _, adv_out = torch.max(adv_out, 1)
            _, ssl_clean_aware_out = torch.max(clean_aware_out, 1)
            _, ssl_adv_aware_out = torch.max(adv_aware_out, 1)
            _, clean_aware_out = torch.max(clean_aware_out, 1)
            _, adv_aware_out = torch.max(adv_aware_out, 1)
            
            running_total_loss += loss
            adv_acc_correct_num = torch.eq(adv_out.cpu().detach(), torch.eye(train_data_size)[targets].argmax(axis=1)).sum()
            adv_correct_sum += adv_acc_correct_num
            

            ssl_clean_correct_num = torch.eq(ssl_clean_aware_out.cpu().detach(), torch.eye(train_data_size)[clean_targets].argmax(axis=1)).sum()
            ssl_adv_correct_num = torch.eq(ssl_adv_aware_out.cpu().detach(), torch.eye(train_data_size)[adv_targets].argmax(axis=1)).sum()
            clean_aware_correct_num = torch.eq(clean_aware_out.cpu().detach(), torch.eye(train_data_size)[targets].argmax(axis=1)).sum()
            adv_aware_correct_num = torch.eq(adv_aware_out.cpu().detach(), torch.eye(train_data_size)[targets].argmax(axis=1)).sum()
            ssl_correct_sum += ssl_clean_correct_num
            ssl_correct_sum += ssl_adv_correct_num
            #print(ssl_correct_sum)
            aware_correct_sum += clean_aware_correct_num
            aware_correct_sum += adv_aware_correct_num
        
        acc = adv_correct_sum.float() / float(train_data_size)
        ssl_acc = ssl_correct_sum.float() / float(2 * train_data_size)
        aware_acc = aware_correct_sum.float() / float(2 * train_data_size)
        running_total_loss = running_total_loss / iter_cnt
        running_adv_clf_loss = running_adv_clf_loss / iter_cnt
        running_ssl_loss = running_ssl_loss / iter_cnt
        running_pr_loss = running_pr_loss / iter_cnt

        tqdm.write('[Epoch %d] Total Loss: %.6f.' % 
                                (epoch, running_total_loss)) # have to be closed zero !
        tqdm.write('[Epoch %d] Training ADV accuracy: %.6f. ADV Clf Loss: %.6f.' % 
                                (epoch, acc, running_total_loss))
        tqdm.write('[Epoch %d] Training SSL accuracy: %.6f. Aware accuracy: %.6f. SSL Loss: %.6f.' % 
                                (epoch, ssl_acc, aware_acc, running_ssl_loss))
        tqdm.write('[Epoch %d] Training Purification Loss: %.6f.' % 
                                (epoch, running_pr_loss))
        #bf1 = acc
        '''
        torch.save({'iter': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),},
                    os.path.join('../checkpoints/feature_extractor', "epoch"+ str(epoch) +"_feature_extractor_acc_"+str(acc)+".pth"))

        torch.save({'iter': epoch,
                    'model_state_dict': clf_head_adv_reweight.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),},
                    os.path.join('../checkpoints/head', "epoch"+ str(epoch) +"_head_acc_"+str(acc)+".pth"))
        tqdm.write('Model saved.')
        '''
        
        model.eval()
        clf_head_adv_reweight.eval()
        
        running_adv_clf_loss = 0.0
        adv_correct_sum = 0
        iter_cnt = 0
        for imgs, targets in tqdm(val_loader) :
            iter_cnt += 1
            #print(imgs)
            #rint(targets)
            targets = np.asarray(targets)
            targets = torch.from_numpy(targets.astype('long'))
            #print(np.shape(targets))
            targets = targets.to(device)
            imgs = imgs.to(device)
            imgs = imgs.float()
            
            ## Adversary generation
            adv_imgs, adv_out, adv_gen_loss= projected_gradient_descent(model, clf_head_adv_reweight, 'val', imgs, targets,
                                cross_entropy, 7, 2.0, 'inf', 0.11, 'inf', val_class_num) #image

            with torch.no_grad():
                #clean_feature = model(imgs)
                adv_feature = model(adv_imgs)

                ## Adversarial-reweighted training
                w = adversarial_reweighting(adv_gen_loss, 7)
                adv_out = clf_head_adv_reweight(adv_feature, 'val')
                w = torch.from_numpy(w.astype('float'))

                loss_ar = adversarial_reweighting_crossentropy(targets, 
                                                                adv_out.cpu(), 
                                                                w, val_class_num)
                #loss_ar = torch.tensor(loss_ar, requires_grad=True)
                running_adv_clf_loss += loss_ar

                _, adv_out = torch.max(adv_out, 1)
                adv_correct_sum += torch.eq(adv_out.cpu().detach(), torch.eye(val_data_size)[targets].argmax(axis=1)).sum()

            val_acc = adv_correct_sum.float() / float(val_data_size)
            val_running_adv_clf_loss = running_adv_clf_loss / iter_cnt

        tqdm.write('[Epoch %d] Validation ADV accuracy: %.4f. ADV Clf Loss: %.3f.' % 
                                (epoch, val_acc, val_running_adv_clf_loss))
        
def get_model(train_class_num, val_class_num, ssl_class_num):

    network = resnet12(avg_pool = True, drop_rate = 0.1, dropblock_size = 5).cuda()    
    head1, head2 = classification_head(train_class_num, val_class_num, ssl_class_num)
    purifier = feature_purification_network().cuda()
    head1 = head1.cuda()
    head2 = head2.cuda()
    
    network = torch.nn.DataParallel(network)
    head1 = torch.nn.DataParallel(head1)
    head2 = torch.nn.DataParallel(head2)
    purifier = torch.nn.DataParallel(purifier)
    print("Model to device ..")
        
    return network, head1, head2, purifier

if __name__=='__main__' :
    print("Go !")
    train()

