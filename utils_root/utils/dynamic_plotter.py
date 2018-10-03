import matplotlib.pyplot as plt


class DynamicPlotter(object):
    def __init__(self, start=True, num=1):
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.magic("matplotlib notebook")

        self.axes = []
        self.fig = plt.figure()
        self.idx = 0
        
        for i in range(num):
            subplot_value = int("{}1{}".format(num, i+1))
            self.axes.append(self.fig.add_subplot(subplot_value))
            
        plt.ion()

        if start:
            self.start()

    def set_index(self, value):
        self.idx = value
        
    def increment_idx(self):
        self.set_index( (self.idx+1) % len(self.axes) )
        
    def start(self):
        self.fig.show()
        self.fig.canvas.draw()

    def stop(self):
        plt.close(self.fig)
        
    def clear_all(self):
        for ax in self.axes:
            ax.clear()
                
    def clear(self):
        self.axes[self.idx].clear()
        
    def draw(self):
        self.fig.canvas.draw()

    def plot(self, data, title=None, save=False):
        self.clear()
        
        self.axes[self.idx].plot(data)
        if title:
            self.axes[self.idx].set_title(title)
           
        self.draw()
        
        if save:
            plt.savefig(save)
        
        self.increment_idx()

    def imshow(self, data, title=None, save=False):
        self.clear()
        
        self.axes[self.idx].imshow(data)
        if title:
            self.axes[self.idx].set_title(title)
                      
        self.draw()
            
        if save:
            plt.savefig(save)
            
        self.increment_idx()

    def show_plot_sample(self):
        from time import sleep
        
        values = []
        for i in range(10):
            values.append(i ** 2)
            self.plot(values)
            sleep(0.1)
            
    def show_image_sample(self):
        from time import sleep
        import numpy as np
        
        for i in range(10):
            self.imshow(np.random.rand(6,6))
            sleep(0.2)
