import numpy as np
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('L', size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

samples = np.load('sample.npy')
processed_array = []

for i in range(samples.shape[0]):
    sample = samples[i, :, :, 0]
    sample = (((sample - sample.min()) / (sample.max() - sample.min())) * 255.9).astype(np.uint8)
    print(np.mean(sample))
    sample = Image.fromarray(sample)
    processed_array.append(sample)

grid = image_grid(processed_array, 4, 5)
grid.save("your_file.jpeg")
