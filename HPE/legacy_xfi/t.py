import numpy as np

file_path = "/apps/users/icps_intelligence/data/lyg/code/MMFi_DATA/E04/S40/A27/rgb/frame297.npy"

data = np.load(file_path)


print(":", file_path)
print(":", data.shape)
print(":", data.dtype)
print("\10 data:")


if data.ndim == 1:

    print(data[:10])
else:
    
    if data.shape[0] > 10:
        print(data[:10])
    else:
        print(data) 