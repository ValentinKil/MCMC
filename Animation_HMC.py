import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def background_plot(ax, target, xmin, xmax, ymin, ymax):
    sns.set_theme(style="white", color_codes=True)
    X, Y = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
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


def update(frame,sample,Trajectories,particle,line,trail):
    current_position = np.array([sample[frame, 0], sample[frame, 1]])

    if sample[frame, 2]:  # Check the additional parameter
        particle.set_offsets(current_position)
        particle.set_facecolor('black')

        # Append the current position to the trail
        trail.set_offsets(np.vstack([trail.get_offsets(), current_position]))

        # Draw the trajectory between the current and previous positions
        if frame > 0 and sample[frame - 1, 2]:
            trajectory = Trajectories[frame - 1]
            line.set_xdata(trajectory[:, 0])
            line.set_ydata(trajectory[:, 1])
            line.set_color('blue')
    else:
        # If the additional parameter is False, use red color
        particle.set_offsets(current_position)
        particle.set_facecolor('red')

        # Draw the trajectory between the current and previous positions
        if frame > 0:
            trajectory = Trajectories[frame - 1]
            line.set_xdata(trajectory[:, 0])
            line.set_ydata(trajectory[:, 1])
            line.set_color('red')

    return particle, trail, line

## MCMC trace plot
def calculate_cumulative_average(sample):
    n_iter = len(sample)
    Cum_sum = np.zeros((n_iter, 2))

    for i in range(n_iter):
        if i == 0:
            Cum_sum[i] = sample[i, 0:2]
        j=i
        while not sample[j,2]:
                j-=1
        Cum_sum[i] = Cum_sum[i - 1] + sample[j, 0:2]
    return Cum_sum / np.arange(1, n_iter + 1)[:, None] # Divide Cum_sum by corresponding index
