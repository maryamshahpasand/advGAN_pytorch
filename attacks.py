import math
import torch
from sklearn import preprocessing
import numpy as np
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(inputs, output):
    x = inputs.squeeze()
    n = inputs.size()[0]
    x = inputs.repeat(output, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(output))
    return x.grad.data


    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    # jacobian= torch.autograd.grad(output ,inputs, retain_graph=True )

    # assert inputs.requires_grad
    #
    # # num_classes = output.size()[1]
    # num_classes = 1
    #
    # jacobian = torch.zeros(num_classes, inputs.size()[1])
    # # jacobian = torch.zeros(1,inputs.size()[1])
    # grad_output = torch.zeros(*output.size())
    # if inputs.is_cuda:
    #     grad_output = grad_output.cuda()
    #     jacobian = jacobian.cuda()
    #
    # for i in range(num_classes):
    #     zero_gradients(inputs)
    #     grad_output.zero_()
    #     grad_output[:, i] = 1
    #     # output.backward(grad_output, retain_variables=True)
    #     output.backward(grad_output)#, retain_graph=True)
    #     # output.backward(grad_output)
    #     jacobian[i] = inputs.grad.data

    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[0]

    jacobian = torch.zeros(inputs.size())
    grad_output = torch.zeros(1)
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    zero_gradients(inputs)
    grad_output.zero_()
    grad_output = 1
    output.backward(grad_output)#, retain_variables=True)  # , retain_graph=True)
    jacobian = inputs.grad.data
    # for i in range(num_classes):
    #     zero_gradients(inputs)
    #     grad_output.zero_()
    #     grad_output[i] = 1
    #     # output.backward(grad_output, retain_variables=True)
    #     output.backward(grad_output)#, retain_graph=True)

        # jacobian[i] = inputs.grad.data

    return jacobian# torch.transpose(jacobian, dim0=0, dim1=1)


    # def get_jacobian(net, x, noutputs):
    # x = inputs.squeeze()
    # n = inputs.size()[0]
    # x = inputs.repeat(output, 1)
    # x.requires_grad_(True)
    # y = net(x)
    # y.backward(torch.eye(output))
    # return x.grad.data


    # return jacobian #torch.transpose(jacobian, dim0=0, dim1=1)


def fgsm(feature,inputs, targets, model, criterion, eps):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
    y = np.load('./MalwareDataset/Drebin_important_features.npy ')
    """
    {'feature':           12, yes
    'permission':         68, yes
    'activity':           60, yes
    'service_receiver':   79, yes
    'provider':            2, yes
    'service':             0,
    'intent':             26, yes
    'api_call':           44,
    'real_permission':    19,
    'call':               20,
    'url':                211}
    """
    y = np.append(y , [y[len(y)-1]], axis=0)
    manifest_features = torch.from_numpy(np.where((y[:, 2] == 'feature') | (y[:, 2] == 'permission') | (y[:, 2] == 'activity') | (
                y[:, 2] == 'service_receiver') | (y[:, 2] == 'provider') | (y[:, 2] == 'service') | (
                         y[:, 2] == 'intent'), np.ones(y.shape[0]), np.zeros(y.shape[0]))).float().cuda()
    code_features =torch.from_numpy(np.where((y[:, 2] == 'api_call') | (y[:, 2] == 'real_permission') | (y[:, 2] == 'call') | (y[:, 2] == 'url'), np.ones(y.shape[0]), np.zeros(y.shape[0]))).float().cuda()
    lb = preprocessing.LabelBinarizer()
    lb.fit([0, 1])
    targets = np.hstack((1 - lb.transform(targets), lb.transform(targets)))
    targets= torch.from_numpy(targets).float().cuda()
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    output = model(crafting_input)
    loss = criterion(output, crafting_target)
    if crafting_input.grad is not None:
        crafting_input.grad.data.zero_()
    loss.backward()
    if feature=='manifest':
        crafting_output = torch.max(crafting_input.data , ((eps*torch.sign(crafting_input.grad.data))*manifest_features))
    elif feature == 'code':
        crafting_output = torch.max(crafting_input.data , ((eps*torch.sign(crafting_input.grad.data))*code_features))
    else:
        crafting_output = torch.max(crafting_input.data , eps*torch.sign(crafting_input.grad.data))
        # print('alllllllllll')

    return torch.abs(crafting_output)


def saliency_map(jacobian, search_space, target_index, increasing=True):
    all_sum = torch.sum(jacobian, 0).squeeze()
    alpha = jacobian[target_index].squeeze()
    beta = all_sum - alpha

    if increasing:
        mask1 = torch.ge(alpha, 0.0)
        mask2 = torch.le(beta, 0.0)
    else:
        mask1 = torch.le(alpha, 0.0)
        mask2 = torch.ge(beta, 0.0)

    mask = torch.mul(torch.mul(mask1, mask2), search_space)

    if increasing:
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    else:
        saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())

    max_value, max_idx = torch.max(saliency_map, dim=0)

    return max_value, max_idx


# TODO: Currently, assuming one sample at each time
def jsma(feature,target_model, sarogate_model, input_tensor, target_class, max_distortion=0.1):
    # y = np.load('./MalwareDataset/X_important.npz')
    y = np.load('./MalwareDataset/Drebin_important_features.npy ')
    """
    {'feature':           12, yes
    'permission':         68, yes
    'activity':           60, yes
    'service_receiver':   79, yes
    'provider':            2, yes
    'service':             0,
    'intent':             26, yes
    'api_call':           44,
    'real_permission':    19,
    'call':               20,
    'url':                211}
    """

    # Make a clone since we will alter the values
    input_features = torch.autograd.Variable(input_tensor.clone(), requires_grad=True)
    # input_features=torch.squeeze(input_tensor)
    num_features = input_features.size()[1]
    max_iter = math.floor(num_features * max_distortion)
    count = 0

    # a mask whose values are one for feature dimensions in search space
    # search_space = torch.ones(num_features).byte()
    # if input_features.is_cuda:
    #     search_space = search_space.cuda()

    output =sarogate_model(input_features)
    source_class = torch.argmax(output)

    while (count < max_iter) and (source_class != target_class): #and (search_space.sum() != 0):
        # Calculate Jacobian
        # print('Calculate Jacobian')
        # print(count)
        jacobian = torch.autograd.grad(output.squeeze()[0], input_features, retain_graph=True)[0]
        # x = input_features.squeeze()
        # n = x.size()[0]
        # x = x.repeat(noutputs, 1)
        # x.requires_grad_(True)
        # y = net(x)
        # y.backward(torch.eye(noutputs))
        # return x.grad.data
        # jacobian = compute_jacobian(input_features, output)

        if torch.argmax(jacobian)<=0:
            # print('no grad')
            break
        # input_features[0][torch.argmax(jacobian)]=1
        # print(torch.argmax(jacobian))
        for i in torch.argsort(jacobian, descending=True)[0]:
            # print(i)
            if input_features[0][i]==0:
                # print("we need to check featue is in permited category")
                if i!= 541: #y has only 541 features
                    # if y[i,2] in ['feature','permission','activity','service_receiver','provider','service','intent']:
                    #check if it is manifest
                    #if y[i,2] in ['feature','permission','activity','service_receiver','provider','service','intent']:
                    #check if it is code
                   # if y[i,2] in ['api_call','real_permission','call','url']:
                        # print(y[i,1])

                    if feature == 'manifest':
                        if y[i, 2] in ['feature', 'permission', 'activity', 'service_receiver', 'provider', 'service',
                                       'intent']:
                            input_features[0][i] = 1
                            break
                    elif feature == 'code':
                        if y[i, 2] in ['api_call', 'real_permission', 'call', 'url']:
                            input_features[0][i] = 1
                            break
                    else:
                        input_features[0][i] = 1
                        break
        # if(torch.sum(jacobian) > 0):
        #     print('hi')

        # increasing_saliency_value, increasing_feature_index = saliency_map(jacobian, search_space, target_class, increasing=True)

        # mask_zero = torch.gt(input_features.data.squeeze(), 0.0)
        # search_space_decreasing = torch.mul(mask_zero, search_space)
        # decreasing_saliency_value, decreasing_feature_index = saliency_map(jacobian, search_space_decreasing, target_class, increasing=False)

        # if increasing_saliency_value == 0.0:# and decreasing_saliency_value == 0.0:
        #     break

        # if increasing_saliency_value > decreasing_saliency_value:
        # input_features.data[0][increasing_feature_index] += 1
        # else:
        # 	input_features.data[0][decreasing_feature_index] -= 1
        # print(count)
        output = sarogate_model(input_features)
        source_class = torch.argmax(output)
        # if source_class == target_class:
        #     print('success with surrogate model with distortion:')
        #     print(torch.sum(input_features - input_tensor))
        source_class = target_model.model.predict(input_features.cpu().detach().numpy())

        count += 1
    # print(count)
    return input_features
