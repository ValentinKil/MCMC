
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def background_plot(ax,target,xmin,xmax,ymin,ymax):
    sns.set_theme(style="white", color_codes=True)
    X, Y = np.mgrid[xmin:xmax:.05, ymin:ymax:.05]
    pos = np.dstack((X, Y))
    
    # Plot the filled contour directly on the ax
    c = ax.pcolormesh(X, Y, target(pos), cmap='Blues', alpha=0.7)
    
    return c

def init(xmin,xmax,ymin,ymax,square,particle,trail,ax):
    ax.set_xlim(xmin, xmax)  # Set your desired x-axis limits
    ax.set_ylim(ymin, ymax)  # Set your desired y-axis limits
    if square:
        plt.axis('square') 
    return particle, trail

def update(frame,sample,particle,line1,line2,trail):
    current_position = np.array([sample[frame, 0], sample[frame, 1]])
  
    particle.set_offsets(current_position)
    particle.set_facecolor('black')

    # Append the current position to the trail
    trail.set_offsets(np.vstack([trail.get_offsets(), current_position]))
        
    # Draw a line between the current and previous positions
    if frame > 0:
        previous_position = np.array([sample[frame-1, 0], sample[frame-1, 1]])
        line1.set_xdata([previous_position[0], current_position[0]])
        line1.set_ydata([previous_position[1], previous_position[1]])
        line2.set_xdata([current_position[0], current_position[0]])
        line2.set_ydata([previous_position[1], current_position[1]])
        line1.set_color('black')
        line2.set_color('black')


    return particle, trail, line1, line2

## MCMC trace plot
def calculate_cumulative_average(sample):
    n_iter = len(sample)
    Cum_sum = np.cumsum(sample,axis=0)
    return Cum_sum / np.arange(1, n_iter + 1)[:, None] # Divide Cum_sum by corresponding index