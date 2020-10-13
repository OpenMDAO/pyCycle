import numpy as np

from matplotlib.path import Path

# defines the verticies of the flight envelope
# [(MN, alt), (MN, alt) ...] 
flt_env_pnts = np.array([[0.0001, -0.00001], 
                          [0.5, 25001], 
                          [1.001, 25001], 
                          [1.001, 0.0],
                          [.00001, -.00001]
                         ])

# matplotlib object that can efficiently test if a point is inside a polygon
flt_env_path = Path(flt_env_pnts)

# helper function to hide the matplotlib syntax 
# (and so I can change it from matplotlib later if I want)
def inside(MN, alt): 
    test_point = np.array([[MN, alt],])
    return flt_env_path.contains_points(test_point)


if __name__ == "__main__": 
    '''
    test script that loops over a structured grid and marks points 
    as inside (blue) or outside (red) the flight envelope
    '''

    import matplotlib.pylab as plt

    fig, ax = plt.subplots()
    ax.plot(flt_env_pnts[:,0], flt_env_pnts[:,1])


    MNs = []
    alts = []
    colors = []
    for MN in np.linspace(0, 1.1, 50): 
        for alt in np.linspace(0, 51000, 50): 
            # test_point = np.array([[MN, alt],])
            color = 'r' 
            # if flt_env_path.contains_points(test_point): 
            if inside(MN, alt): 
                color = 'C0'
            
            colors.append(color)
            MNs.append(MN)
            alts.append(alt)
            
    plt.scatter(MNs, alts, c=colors, s=.05)
    
    plt.show()





