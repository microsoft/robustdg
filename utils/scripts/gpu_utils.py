try: import pycuda.driver as cuda
except: print ("pycuda not available")

import torch
import sys, os, glob, subprocess

def get_gpu_info(print_info=True, get_specs=False):
    cuda.init()
    if get_specs: gpu_specs = cuda.Device(0).get_attributes() # assume same for all (dnnx)
    else: gpu_specs = None

    gpu_info = {
        'available': torch.cuda.is_available(),
        'num_devices': torch.cuda.device_count(),
        'devices': set([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]),
        'current device id': torch.cuda.current_device(),
        'allocated memory': torch.cuda.memory_allocated(),
        'cached memory': torch.cuda.memory_cached()
    }

    if print_info:
        for k,v in gpu_info.items(): print ("{}: {}".format(k, v))

    return gpu_info, gpu_specs


def get_device(device_id=None): # None -> cpu
    device = 'cuda:{}'.format(device_id) if device_id is not None else 'cpu'
    device = torch.device(device if torch.cuda.is_available() and device_id is not None else 'cpu')
    return device

def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

def get_cuda_version():
    """Get CUDA version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\version.txt"
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        path = '/usr/local/cuda/version.txt'
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n','')
        return data
    else:
        return "No CUDA in this machine"

def get_cudnn_version():
    """Get CUDNN version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #candidates = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\include\\cudnn.h"]
    elif sys.platform == 'linux':
        candidates = ['/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h',
                      '/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    elif sys.platform == 'darwin':
        candidates = ['/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    for c in candidates:
        file = glob.glob(c)
        if file: break
    if file:
        with open(file[0], 'r') as f:
            version = ''
            for line in f:
                if "#define CUDNN_MAJOR" in line:
                    version = line.split()[-1]
                if "#define CUDNN_MINOR" in line:
                    version += '.' + line.split()[-1]
                if "#define CUDNN_PATCHLEVEL" in line:
                    version += '.' + line.split()[-1]
        if version:
            return version
        else:
            return "Cannot find CUDNN version"
    else:
        return "No CUDNN in this machine"

if __name__=='__main__':
    print ('gpu name', get_gpu_name())
    print ('cuda', get_cuda_version())
    print ('cudnn', get_cudnn_version())
    print ('device0', get_device(0))
    print ('available', torch.cuda.is_available())