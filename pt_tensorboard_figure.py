from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# create summary writer
writer = SummaryWriter('lightning_logs')

# write dummy image to tensorboard
img_batch = np.zeros((16, 3, 100, 100))
writer.add_images('my_image_batch', img_batch, 0)

# write dummy figure to tensorboard
plt.imshow(np.transpose(img_batch[0], [1, 2, 0]))
plt.title('example title')
writer.add_figure('my_figure_batch', plt.gcf(), 0)
writer.close()
