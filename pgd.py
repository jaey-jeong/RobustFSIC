import torch
import torchvision
import numpy as np

from utils.utils import get_adv_cross_entropy

def projected_gradient_descent(model, clf_head, mode, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm,
                               class_num, clamp=(0,8), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]    
    flag = True

    for _ in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        feature = model(_x_adv)
        feature = feature.cuda()
        adv_out = clf_head(feature, mode)
        loss = loss_fn(y.cpu(), adv_out.cpu(), class_num)
        
        loss_ar = get_adv_cross_entropy(y.cpu(), adv_out.cpu(), class_num)

        if flag :
            v = loss_ar
            flag = False      
        else :      
            v = torch.cat((v, loss_ar), axis = 1)
        
        #loss = torch.tensor(loss, requires_grad = True)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                #print(_x_adv.grad)
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                    .norm(step_norm, dim=-1)\
                    .view(-1, num_channels, 1, 1)

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta
        
        x_adv = x_adv.clamp(*clamp)
    
        #torchvision.utils.save_image(x_adv[0], 'adv'+str(i)+'.jpg')

    return x_adv.detach(), adv_out, v