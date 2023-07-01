import numpy as np
import imageio
sample = np.load('sample.npy')
sample = sample[-2, :, :, 0]
sample = (((sample - sample.min()) / (sample.max() - sample.min())) * 255.9).astype(np.uint8)


print(np.max(sample))
print(np.min(sample))
print(np.mean(sample))
print(sample.shape)
from PIL import Image
im = Image.fromarray(sample)

im.save("your_file.jpeg")
# imageio.imwrite('astronaut-gray.jpg', sample)