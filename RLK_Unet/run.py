import os
import torch

from RLK_Unet.data_process import data_loaders
from RLK_Unet.network_architecture import RLKunet
from RLK_Unet.utils import numpy_convert, label_thr, save_prediction


def cuda_device(GPU_num):
    torch.cuda.is_available()
    device = torch.device(f'cuda:{GPU_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) 
    print ('Current cuda device Number =', torch.cuda.current_device())

    if device.type == 'cuda':
        print('Current cuda device Name =', torch.cuda.get_device_name(GPU_num))
        print('Current cuda device Memory = {:.2f}GB'.format((torch.cuda.get_device_properties(GPU_num).total_memory) * 1e-9))
    return device


def test_process(input_image, num, weight_path, device):
    print("Predicting {}-model ...".format(num))
    weight  = weight_path + '/RLK_Unet_'+str(num)+'.model'

    net     = RLKunet()
    net.to(device)

    net.load_state_dict(torch.load(weight))
    net.eval()
    _, _, _, output = net(input_image)
    output  = output[:,0,:,:,:]
    output  = torch.unsqueeze(output,1)
    output  = torch.where(output>=0.5, 1.0, 0.0)
    return output


def run_model(input_img, output_result, weight_path, GPU_num=0, result_thr=3):
    ### CUDA DEVICE
    device = cuda_device(GPU_num)

    ### DATA LOADER
    loader_data     = data_loaders(img_data=input_img)

    ### MODEL RUNNING
    with torch.no_grad():
        for i, test_data in enumerate(loader_data):
            img         = test_data
            img         = img.to(device).float()

            output0     = test_process(img, 0, weight_path, device)
            output1     = test_process(img, 1, weight_path, device)
            output2     = test_process(img, 2, weight_path, device)
            output3     = test_process(img, 3, weight_path, device)
            output4     = test_process(img, 4, weight_path, device)

            print("\nPost-processing ...")
            outputs     = output0 + output1 + output2 + output3 + output4
            outputs     = torch.where(outputs>2.5, 1, 0)
            # thr label
            outputs  = label_thr(outputs, result_thr).to(device)

            # save nifti
            print("Saving ...")
            save_prediction(
                            results=numpy_convert(outputs), 
                            row_data=input_img, 
                            output_path=str(output_result),
                            name=os.path.basename(input_img)
                            )
            print("COMPLETE !")
            print("")
            print("")
