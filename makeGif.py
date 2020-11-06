import imageio
import tfutils

path = r'C:\Users\samed\OneDrive\Bilder\AI'
gifpath = r'C:\Users\samed\OneDrive\Bilder\Gifs'
file = r'\movie1.gif'
filenames = tfutils.retdir(path)

with imageio.get_writer(gifpath+file, mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)