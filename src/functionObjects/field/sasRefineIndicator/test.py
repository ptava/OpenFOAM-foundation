import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import pow, exp

# rc_fonts = {
#     "text.usetex": True,
#     'mathtext.default': 'regular',
#     'text.latex.preamble': [r"""\usepackage{bm}"""],
# }
# mpl.rcParams.update(rc_fonts)

# Plot function f(d) = d * (1 + omega * (1 - exp(-invTwoSigma * sqr(d))));
# invTwoSigma = 0.5 * sqrt(sigma)
# con -100 < d < 100

# Designed to be used with limit cases defined by:
# - sigma = 1.0 and omega = 2.0
# - sigma = 1.0 and omega = 1.0
def plot_core_function(axs, d, omega, sigma, function_name):
    alpha = 0.6
    dMax = np.max(d)
    # Total shifted distance at the right boundary
    dMaxShift = dMax + dMax * omega
    dShift = d + dMax * omega
    
    # 1. Define the Saturation (Soft-Core) component
    def get_S(x):
        # Using 0.5/omega^2 to match your original invTwoSigma logic
        return 1 - np.exp(-0.5 * x**2 / sigma**2)


    k = get_S(dMaxShift) / (dMaxShift * alpha) - 1

    # 4. Final Function Construction
    # We use d_pos = dShift for the decay to ensure the peak is relative 
    # to the shifted origin.
    d_pos = np.maximum(0, dShift)
    envelope = 1 / (1 + k * d_pos**2)
    
    f_d = dShift * get_S(dShift) * envelope


    f_d /= np.max(f_d)  # Normalize to max value 1
    axs.plot(d, f_d, label=f'weight={omega}, sigma={sigma}')
    axs.plot(d, envelope)
    axs.set_title(f'Transfer Function: {function_name}', fontweight='bold')
    axs.set_xlabel('d', fontweight='bold')
    axs.set_ylabel(r'$\mathbf{f(d)}$')
    axs.legend()

def plot_periphery_function(axs, nLvk, weight, sigma, function_name):
    invTwoSigma = 0.5 / pow(sigma,2)
    f_d = np.exp(-invTwoSigma * (nLvk-1)**2) - weight*(nLvk-1)**2;
    axs.plot(nLvk, f_d, label=f'weight={weight}, sigma={sigma}')
    axs.set_title(f'Transfer Function: {function_name}', fontweight='bold')
    axs.set_xlabel(r'$\mathbf{\hat{Lvk}}$')
    axs.set_ylabel(r'$\mathbf{f(\hat{Lvk})}$')
    axs.set_ylim([-4.3, 1.3])
    axs.legend()

# Data
transferFunctions = ['oddScaler', 'gaussSink']
d = np.linspace(-3, 3, 400)
coreWeights = [1]
sigmas = [0.1]
peripheryWeights1 = [0.1, 1, 10]
nLvk = np.linspace(1, 2, 400)

def main():
    # create subplots for the number of elements in transferFunctions
    fig, axs = plt.subplots(len(transferFunctions), 1, figsize=(8, 6 * len(transferFunctions)))

    # first subplot
    for omega in coreWeights:
        for sigma in sigmas:
            plot_core_function(axs[0], d, omega, sigma, transferFunctions[0])

    # second subplot
    for weight in peripheryWeights1:
        for sigma in sigmas:
            plot_periphery_function(axs[1], nLvk, weight, sigma, transferFunctions[1])

    minor_ticks_y = np.arange(-2, 1.2, 0.5)
    major_ticks_y = np.arange(-2, 1.2, 1)
    axs[1].set_yticks(major_ticks_y)
    axs[1].set_yticks(minor_ticks_y, minor=True)

    minor_ticks_x = np.arange(0, 4, 0.1)
    major_ticks_x = np.arange(0, 4.1, 0.2)
    axs[1].set_xticks(major_ticks_x)
    axs[1].set_xticks(minor_ticks_x, minor=True)
    
    axs[1].grid(visible=True, which='major', axis='both', color='lightgray')
    axs[1].grid(visible=True, which='minor', axis='both', color='lightgray', linestyle='--')

    axs[1].plot([0, 1], [1, 1], color='C0')
    axs[1].set_xlim([0, 2])
    axs[1].set_ylim([-2, 1.2])


    minor_ticks_y = np.arange(-1, 1.1, 0.1)
    major_ticks_y = np.arange(-1, 1.1, 0.2)
    axs[0].set_yticks(major_ticks_y)
    axs[0].set_yticks(minor_ticks_y, minor=True)
    minor_ticks_x = np.arange(-3, 3.1, 0.5)
    major_ticks_x = np.arange(-3, 3.1, 1)
    axs[0].set_xticks(major_ticks_x)
    axs[0].set_xticks(minor_ticks_x, minor=True)

    axs[0].grid(visible=True, which='major', axis='both', color='lightgray')
    axs[0].grid(visible=True, which='minor', axis='both', color='lightgray', linestyle='--')

    # put a vertical and horizontal line at 0
    axs[0].axvline(0, color='black', linewidth=0.5)
    axs[0].axhline(0, color='black', linewidth=0.5)


    axs[0].annotate("",
            xy=(-1.5, 0), xycoords='data',
            xytext=(-1.2, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="<|-",
                            connectionstyle="angle, angleA=0, angleB=90, rad=0",
                            color="C2")
    )
    axs[0].annotate("",
            xy=(-0.9999, 0), xycoords='data',
            xytext=(-0.6, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="<|-",
                            connectionstyle="angle, angleA=0, angleB=90, rad=0",
                            color="C1")
    )
    axs[0].annotate("",
            xy=(0, 0), xycoords='data',
            xytext=(0.3, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="<|-",
                            connectionstyle="angle, angleA=0, angleB=90, rad=0",
                            color="C0")
    )



    plt.show()

if __name__ == "__main__":
    main()
