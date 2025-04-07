"""Plot function for the QuadWire model"""

# %% import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from matplotlib.patches import Polygon
import os, shutil



# %% plot function

def plotMesh(axis, L, x, elems, edgecolor='black', color='white', clrfun=[], outer=True):
    """
    Mesh display function using parallelepiped elements with triangular facets.
    The colormap can either display quantities expressed at the particules or expressed at the elements

    Parameters
    ----------
    axis : matplotlib axes 3D 
        axis of the matplotlib 3D plot.
    L : int 
        Length of the wire in mm.
    x : array of size (nNodesTOT, nCoord, nParticules)
        Coordinates of each particule of each node of the mesh.
    elems : array of size ((nNodes - 1)*nLayers, elemOrder +1 ) or (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an element.
    edgecolor : str, optional
        Color of the edges of the triangles. The default is 'black'.
    color : str, optional
        Color of the edges of the triangles. The default is 'white'.
    clrfun : array, optional
        Array giving the value that will be used for the colormap. 
        It can either be expressed at each element or at each particule.
        The default is [].
    outer : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    srf : matplotlib 3D polyline
        Plotted surface.
    tri : array of shape (nElemsTot*12, 3)
        Colormap value at each node for each triangle (12 per element) 

    """
    x = np.moveaxis(x, (0, 1, 2), (0, 2, 1)).reshape(4 * x.shape[0], 3)
    # ax.scatter(x[:,0,:].flatten(), x[:,1,:].flatten(), x[:,2,:].flatten())
    # faces are quadrilaterals with outgoing normals (corners defined in the trigo direction)
    lf = 4 * elems[:, [0, 1, 1, 0]] + [2, 2, 0, 0]  # LEFT faces
    rf = 4 * elems[:, [1, 0, 0, 1]] + [3, 3, 1, 1]  # RIGHT faces
    tf = 4 * elems[:, [0, 1, 1, 0]] + [0, 0, 1, 1]  # TOP faces
    bf = 4 * elems[:, [1, 0, 0, 1]] + [3, 3, 2, 2]  # BOTTOM faces
    ff = 4 * elems[:, [1, 1, 1, 1]] + [2, 3, 1, 0]  # FRONT faces
    bkf = 4 * elems[:, [0, 0, 0, 0]] + [3, 2, 0, 1]  # BACK faces
    faces = np.vstack((lf, rf, tf, bf, ff, bkf))  # )) #
    if outer:  # show only outer faces
        tol = L * 1e-6
        ux, idx = np.unique(np.round(x / tol), axis=0, return_inverse=True)
        ufaces = idx[faces]

        f2n = sp.sparse.csr_matrix((np.ones(faces.size), (ufaces.flatten(), np.repeat(np.arange(faces.shape[0]), 4, axis=0))))  # node-face connectivity

        f2f = f2n.T @ f2n  # face-face connectivity
        isinner = ((f2f - 4 * sp.sparse.eye(faces.shape[0])) == 4).sum(axis=1).astype(bool)
        faces = faces[np.invert(np.array(isinner).flatten()), :]

    # conversion to a triangulation (matplotlib limitation)
    tri = np.vstack((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
    srf = axis.plot_trisurf(x[:, 0].flatten(), x[:, 1].flatten(), x[:, 2].flatten()
                          , triangles=tri
                          , edgecolor=edgecolor
                          , color=color
                          , linewidth=0.1
                          , antialiased=True
                          )  # , alpha = .5         , vmax = max(clrfun)          , cmap="OrRd" , cmap = 'Spectral_r'

    # colored faces based on a color function
    if np.array(clrfun).size != 0:
        if np.array(clrfun).size == x.shape[0]:
            # print(np.mean(clrfun[tri], axis=1).shape)
            srf.set_array(np.mean(clrfun[tri], axis=1))
        else:
            if outer:
                clrfun = np.tile(clrfun, 6 * 2)[np.invert(np.array(isinner).flatten()), :]
                # print("trial shape " +str(trial.shape))
                srf.set_array(clrfun)
            else:
                clrfun = np.tile(clrfun, 6 * 2)
                # print("trial shape " +str(trial.shape))
                srf.set_array(clrfun)

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])

    return srf, tri

def usePlotMesh(meshing, X, U, Elems, scale, scfplot, clrfun=[], toPlot=True, elemOrder=1):
    L, Hn, Hb, *_ = meshing
    from modules import mesh, fem
    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)

    # mean basis vectors on nodes
    Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
    nm = Me @ n
    nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
    bm = Me @ b
    bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]

    # undeformed shape
    x0 = X[:, :, np.newaxis] + 0.5 * (
            Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
        [[[1, 1, -1, -1]]]))
    # deformed shape
    uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
    x = x0 + scale * uplot

    if toPlot:
        fig = plt.figure()
        ax = plt.axes(projection='3d', proj_type='ortho')
        ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2])*scfplot))
    else:
        # Create figure without displaying
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    srf, tri = plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clrfun, outer=False)

    if toPlot:
        colorbar = plt.colorbar(srf, pad=0.15)
        plt.show()
    else:
        plt.close(fig)  # Clean up the temporary figure

    return srf, tri

def plot_data(x, U, data, Hn, Hb, L, Elems, scfplot=1, title='Title', xlbl='Axis t', ylbl='Axis n', zlbl='Axis b'):
    """
    Plots data on the provided figure using plotMesh and creates a colorbar.

    Args:
      fig: Matplotlib figure object.
      data: Data points for the mesh.
      clr: Color function for the data.
      title: Title for the plot.
    """

    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type='ortho')
    # ax.set_axis_off()

    # plotMesh(x,Elems,color='none',edgecolor='gray')
    ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scfplot), np.ptp(x[:, 2]) * scfplot))

    # deformed shape
    scale = .25 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2))
    uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
    x = x + scale * uplot

    clr = data
    srf, tri = plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
    colorbar = plt.colorbar(srf, pad=0.15)
    colorbar.set_label('clr')

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.set_title(title)

    plt.show()


def plot_sigma(Sigma, qp2elem, nQP, x, Elems, L, Hn, Hb, field, scfplot):
    """
    field can be stt stn stb
    """
    if field == 'stt':
        projection_index = 0
        label = '$\sigma_{tt}$'
    elif field == 'stn':
        projection_index = 3
        label = '$\sigma_{tn}$'
    elif field == 'stb':
        projection_index = 4
        label = '$\sigma_{tb}$'
    else:
        return 'field type is unknown, should be among stt stn stb'

    sigmaplot = qp2elem @ Sigma[projection_index * nQP:(projection_index + 1) * nQP]
    clr = sigmaplot[:, 0]
    clr = clr / (Hn * Hb)

    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type='ortho')
    ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scfplot), np.ptp(x[:, 2] * scfplot)))
    #ax.set_title(zigzag * 'ZigZag' + (1 - zigzag) * 'ZigZig')

    srf, tri = plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
    colorbar = plt.colorbar(srf, pad=0.15)
    colorbar.set_label(f'{label} [MPa]')


def simpleplot(x, y, xlbl, ylbl, clr='black', grid=True):
    plt.figure()
    plt.plot(x, y, color=clr)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if grid:
        plt.grid()
    plt.show()

#%% plot trajectory (zigzag, offset...)

def plotpoints(X, label=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('t Axis')
    ax.set_ylabel('n Axis')
    ax.set_zlabel('b Axis')
    # Tracer les points en 3D
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o')
    if label :
        for node in range(len(X)):
            ax.text(X[node, 0], X[node, 1], X[node, 2], node)
    plt.show()

def plotnormal(X, n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('t Axis')
    ax.set_ylabel('n Axis')
    ax.set_zlabel('b Axis')
    # Tracer les points en 3D
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o')
    ax.scatter(n[:, 0], n[:, 1], n[:, 2], c='g', marker='o')
    plt.show()

def spyplt(M, ttl=''):
    plt.spy(M, markersize=1), plt.title(ttl), plt.show()


# %% plotting Sw

def plotmatrix(M, ttl=''):
    """
    Display matrix with intensity color using the diverging colormap PiYG.
    Center the colormap around zero.
    """
    fig, ax = plt.subplots()

    # Find the maximum absolute value in the matrix
    max_abs_value = max(abs(M.min()), abs(M.max()))

    # Get the original PiYG colormap
    base_cmap = plt.cm.PiYG

    # Create a new colormap based on PiYG with a distinct color for value 1
    # Extract colors from PiYG and modify around the middle
    colors = base_cmap(np.linspace(0, 1, 256))

    # Make the color for the value around 0 more distinct
    highlight_color = [0.8, 0.8, 0.8, 1] # light grey (RGBA) #[1, 1, 0.5, 1]  # Light yellow (RGBA)
    colors[128] = highlight_color  # Adjust color at the center for the value 0

    # Create a new colormap with modified colors
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_PiYG', colors)

    # Normalize the color range around zero
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)

    # Plot matrix with pcolormesh
    c = plt.pcolormesh(M, cmap=custom_cmap, edgecolors='lightgrey', linewidth=0.1, norm=norm)
    ax = plt.gca()
    plt.gca().invert_yaxis()

    ax.set_aspect('equal')
    ax.set_title(ttl)

    plt.colorbar(c, ax=ax)

    plt.show()

def plotmatrix_sparse_t(M, nNodesTot, ttl=''):
    """
    plot t block of sparse matrix M
    """
    plotmatrix(M.todense()[:4 * nNodesTot, :4 * nNodesTot], ttl)

def plotmatrix_t(M, nNodesTot, ttl=''):
    """
    plot t block of numpy matrix M
    """
    plotmatrix(M[:4 * nNodesTot, :4 * nNodesTot], ttl)

def plotmatrix_sparse_t_evenNodes(M, nNodesTot, ttl=''):
    """
    plot t block of sparse matrix M for even nodes (requiring even nodes per bead)
    """
    print(M.todense()[:4 * nNodesTot, :4 * nNodesTot][::2, ::2])
    plotmatrix(M.todense()[:4 * nNodesTot, :4 * nNodesTot][::2, ::2], ttl)

def plotmatrix_t_evenNodes(M, nNodesTot, ttl=''):
    """
    plot t block of numpy matrix M for even nodes (requiring even nodes per bead)
    """
    print(M[:4 * nNodesTot, :4 * nNodesTot][::2, ::2])
    plotmatrix(M[:4 * nNodesTot, :4 * nNodesTot][::2, ::2], ttl)


# %% plotting optim results
def plotparam(param_values, ttl='', gap=5, difference=False):
    param_values_array = np.array(param_values)
    if difference :
        param_values_array -= param_values_array[0] # difference par rapport à l'état initial
    nLayers_v = len(param_values_array[0])
    num_sets = len(param_values_array)
    # Use a colormap to generate colors for the intermediate sets
    cmap = plt.get_cmap('PuRd')
    colors = cmap(np.linspace(0, 1, num_sets - 2))
    color_firstLayer = 'magenta'
    color_lastLayer = 'navy'

    # Create line plot
    plt.figure(figsize=(10, 8))

    # Plot the first set with a distinct style
    plt.plot(param_values_array[0], np.arange(nLayers_v), label='k=0 (Start)', color=color_firstLayer, linewidth=2.5, linestyle='--', marker='o')

    # Plot the intermediate sets, and add rectangles around each data point
    for i, (row, color) in enumerate(zip(param_values_array[1:-1], colors), start=1):
        plt.plot(row, np.arange(nLayers_v), label=f'k={i}'*(i%gap==0), color=color, linestyle=':', marker='o')            # (i%5==0) only label every 5 iterations
        if not difference :
            for layer in range(nLayers_v):
                rect = Rectangle((param_values_array[i][layer]-1/2, layer-1/2), 1, 1, linewidth=0.5, edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)

    # Plot the first set again without legend to make it more visible, and add rectangles around each data point
    plt.plot(param_values_array[0], np.arange(nLayers_v), color=color_firstLayer, linewidth=2.5, linestyle='--', marker='o')
    if not difference :
        for layer in range(nLayers_v):
            rect = Rectangle((param_values_array[0][layer]-1/2, layer-1/2), 1, 1, linewidth=1.2, edgecolor=color_firstLayer, facecolor='none')
            plt.gca().add_patch(rect)

    # Plot the last set with a distinct style, and add rectangles around each data point
    plt.plot(param_values_array[-1], range(len(param_values_array[-1])), label=f'k={num_sets - 1} (End)', color=color_lastLayer, linewidth=2.5, linestyle='--', marker='o')
    if not difference :
        for layer in range(nLayers_v):
            rect = Rectangle((param_values_array[-1][layer]-1/2, layer-1/2), 1, 1, linewidth=1.2, edgecolor=color_lastLayer, facecolor='none')
            plt.gca().add_patch(rect)

    # Set integer ticks for the y-axis
    plt.yticks(np.arange(nLayers_v))

    # Set the aspect ratio of the plot
    aspectratio = 0.1/0.2                 # make rectangles slimmer, ie Hb/Hn ratio
    plt.gca().set_aspect(aspectratio)

    plt.title(ttl)
    plt.xlabel('Offset')
    plt.ylabel('Layers')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plotvalues(J_values, ttl=''):
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('PuRd')
    colors = cmap(np.linspace(0, 1, len(J_values)))


    if len(np.array(J_values).shape) == 1:
        plt.plot(J_values, linestyle=':', marker='+')
    else :
        for k in range(len(J_values)) :
            plt.plot(J_values[k], label=f'k={k}', color=colors, linestyle=':', marker='+')
        plt.legend(loc='best')

    plt.title(ttl)
    plt.xlabel('Iterations')
    plt.ylabel('')

    #plt.xscale('log')  # Set x-axis scale to logarithmic
    plt.yscale('log')  # Set y-axis scale to logarithmic

    plt.grid(True)
    plt.show()




# %% poc

def save_gif(output_dir, image_filenames, gif_filename="plots_animation.gif", duration=500, loop=0, start=True):

    # Combine output_dir and gif_filename to create full path for saving the GIF
    gif_filepath = os.path.join(output_dir, gif_filename)

    # Create a list to store the image objects
    images = [Image.open(img) for img in image_filenames]

    # Save as a GIF
    images[0].save(gif_filepath, save_all=True, append_images=images[1:], duration=duration, loop=loop)

    # Open the folder with the saved images
    os.startfile(output_dir)

    # Optionally open the GIF file itself
    if start:
        os.startfile(gif_filepath)

    return gif_filepath


def plot_U0_layer_n(n, title, meshing, U, split_bool=False):
    """
    Plot Ut and Un for particle 0 of the first bead of every layer with optional y-axis limits and padding.

    Arguments:
    meshing     : tuple - Contains meshing information: (L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes).
    U           : numpy array - The data array for displacement, shape (3, 4, nNodesTot).
    split_bool  : bool (optional) - Whether to account for bead splitting, default is False.
    ylim        : tuple (optional) - Predefined y-limits ((ymin_t, ymax_t), (ymin_n, ymax_n)), default is None.
    padding     : float (optional) - Fraction of the data range to add as padding, default is 0.1 (10% padding).

    Returns:
    -------
    ylim : tuple - Returns y-limits ((ymin_t, ymax_t), (ymin_n, ymax_n)) after plotting if none is provided.
    """
    L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes = meshing
    nNodes_h = (nLayers_h + 1 * split_bool) * nNodes

    # Create a figure with two subplots
    fig, (ax_t, ax_n) = plt.subplots(2, 1, figsize=(14, 10))  # Two rows, one column
    # Create a container for legend handles and labels
    handles = []
    labels = []

    # Extract the data for U_t and U_n
    U_t_data = U[0, 0, (n-1) * nNodes_h : n * nNodes_h]
    U_n_data = U[1, 0, (n-1) * nNodes_h : n * nNodes_h]

    # Plot U_t on the first subplot (ax1)
    line1, = ax_t.plot(U_t_data, label=f'Bead {n-1}')

    # Plot U_n on the second subplot (ax2)
    line2, = ax_n.plot(U_n_data)

    # Append the handle and label for the legend
    handles.append(line1)
    labels.append(f'Bead {n-1}')

    # Customize the first subplot (U_t)
    ax_t.set_xlabel('t')
    ax_t.set_ylabel('$U_t (mm)$')

    # Customize the second subplot (U_n)
    ax_n.set_xlabel('t')
    ax_n.set_ylabel('$U_n (mm)$')

    fig.suptitle(title+f'thinwall of {nLayers_v} printed layers')

    # Create a shared legend outside the figure
    plt.subplots_adjust(right=0.8)  # Leave 20% space for the legend
    #fig.legend(handles, labels, title='top left particle', bbox_to_anchor=(0.90, 0.5), loc='center')

    # Display the combined plot
    plt.show()

    return

def plot_U0_layers(title, meshing, U, nLayers_v_min=0, nLayers_v_max=15, split_bool=False, ylim=None, padding=0.1):
    """
    Plot Ut and Un for particle 0 of the first bead of every layer with optional y-axis limits and padding.

    Arguments:
    meshing     : tuple - Contains meshing information: (L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes).
    U           : numpy array - The data array for displacement, shape (3, 4, nNodesTot).
    split_bool  : bool (optional) - Whether to account for bead splitting, default is False.
    ylim        : tuple (optional) - Predefined y-limits ((ymin_t, ymax_t), (ymin_n, ymax_n)), default is None.
    padding     : float (optional) - Fraction of the data range to add as padding, default is 0.1 (10% padding).

    Returns:
    -------
    ylim : tuple - Returns y-limits ((ymin_t, ymax_t), (ymin_n, ymax_n)) after plotting if none is provided.
    """
    L, Hn, Hb, beadType, layerType, nLayers_h, nLayers_v, nNodes = meshing
    nNodes_h = (nLayers_h + 1 * split_bool) * nNodes

    # Create a figure with two subplots
    fig, (ax_t, ax_n) = plt.subplots(2, 1, figsize=(14, 10))  # Two rows, one column

    # Choose a colormap
    cmap = plt.cm.plasma

    # Normalize the layer index to the colormap
    colors = cmap(np.arange(1, max(nLayers_v, nLayers_v_max)+1)/max(nLayers_v, nLayers_v_max))

    # Create a container for legend handles and labels
    handles = []
    labels = []

    # Variables to track y-limits
    ymin_t, ymax_t = float('inf'), float('-inf')
    ymin_n, ymax_n = float('inf'), float('-inf')

    # Loop through each layer and plot on both subplots
    for n in range(np.min([nLayers_v_min, nLayers_v])+1, np.min([nLayers_v_max, nLayers_v])+1):
        print(n)
        # Extract the data for U_t and U_n
        U_t_data = U[0, 0, (n-1) * nNodes_h : n * nNodes_h]
        U_n_data = U[1, 0, (n-1) * nNodes_h : n * nNodes_h]

        # Plot U_t on the first subplot (ax1)
        line1, = ax_t.plot(U_t_data, label=f'Bead {n-1}', color=colors[n-1])
        ymin_t = min(ymin_t, np.min(U_t_data))
        ymax_t = max(ymax_t, np.max(U_t_data))

        # Plot U_n on the second subplot (ax2)
        line2, = ax_n.plot(U_n_data, color=colors[n-1])
        ymin_n = min(ymin_n, np.min(U_n_data))
        ymax_n = max(ymax_n, np.max(U_n_data))

        # Append the handle and label for the legend
        handles.append(line1)
        labels.append(f'Bead {n-1}')

    # Add padding to the calculated y-limits
    if ylim is None:
        range_t = ymax_t - ymin_t
        range_n = ymax_n - ymin_n

        ymin_t -= padding * range_t
        ymax_t += padding * range_t
        ymin_n -= padding * range_n
        ymax_n += padding * range_n
        ylim = ((ymin_t, ymax_t), (ymin_n, ymax_n))
    else:
        (ymin_t, ymax_t), (ymin_n, ymax_n) = ylim

    # Set the y-limits with the added padding
    ax_t.set_ylim(ymin_t, ymax_t)
    ax_n.set_ylim(ymin_n, ymax_n)

    # Customize the first subplot (U_t)
    ax_t.set_xlabel('t')
    ax_t.set_ylabel('$U_t (mm)$')

    # Customize the second subplot (U_n)
    ax_n.set_xlabel('t')
    ax_n.set_ylabel('$U_n (mm)$')

    fig.suptitle(title+f'thinwall of {nLayers_v} printed layers')

    # Create a shared legend outside the figure
    plt.subplots_adjust(right=0.8)  # Leave 20% space for the legend
    fig.legend(handles, labels, title='top left particle', bbox_to_anchor=(0.90, 0.5), loc='center')

    # Display the combined plot
    plt.show()

    return ylim

def plot_cross_section(ax, cross_section_x, clrmap=plt.cm.plasma, legend=True, ylim=None, padding=0.1):
    """
    Plot the cross-section of the beads based on corner coordinates for all layers at once.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the cross-section.

    cross_section_x : np.array of shape (nLayers, 3, 4)
        Coordinates of 4 bead corners for each layer, with 3 coordinates for each corner.

    clrmap : colormap, optional
        Colormap to use for the layers, default is plt.cm.plasma.

    legend : bool, optional
        Whether to display a legend for the layers, default is True.

    ylim : tuple, optional
        Predefined y-limits (ymin, ymax), default is None.

    padding : float, optional
        Fraction of the data range to add as padding, default is 0.1 (10% padding).
    """
    nLayers_v = cross_section_x.shape[0]
    # Extract x, y coordinates for all layers
    x_coords = cross_section_x[:, 1, :]  # x-coordinates for all layers
    y_coords = cross_section_x[:, 2, :]  # y-coordinates for all layers
    # Close the loop for each layer by appending the first corner to the end
    x_coords = np.hstack((x_coords, x_coords[:, [0]]))  # Add first column to end
    y_coords = np.hstack((y_coords, y_coords[:, [0]]))  # Add first column to end

    # Define colormap and fixed colors
    cmap = clrmap  # plt.cm.plasma or plt.cm.Greys
    colors = [cmap(c / nLayers_v) for c in range(1,nLayers_v+1)]

    # Variables to track y-limits
    ymin = float('inf')
    ymax = float('-inf')

    # Loop through each set of corners and plot the corresponding rectangle
    for i in range(x_coords.shape[0]):
        # Define the vertices for the rectangle in the given order
        vertices = np.array([
            [x_coords[i, 0], y_coords[i, 0]],  # Top Left
            [x_coords[i, 1], y_coords[i, 1]],  # Top Right
            [x_coords[i, 3], y_coords[i, 3]],  # Bottom Right
            [x_coords[i, 2], y_coords[i, 2]],  # Bottom Left
        ])

        # Create a rectangle (Polygon) using the vertices
        polygon = Polygon(vertices, closed=True, edgecolor=colors[i], facecolor='none', label=f'Bead {i}')

        # Add the rectangle to the plot
        ax.add_patch(polygon)

        # Update y-limits
        ymin = min(ymin, y_coords[i].min())
        ymax = max(ymax, y_coords[i].max())

        # # Add label for each layer
        # if legend:
        #     ax.text(np.mean(x_coords[i, :]), np.mean(y_coords[i, :]), f'{i}',
        #             fontsize=10, ha='center', va='center', color=colors[i])

    if legend:
        plt.legend()

    # Set plot limits to fit all rectangles, apply padding if not provided
    if ylim is None:
        ax.set_xlim(x_coords.min() - 1, x_coords.max() + 1)
        ax.set_ylim(ymin - padding * (ymax - ymin), ymax + padding * (ymax - ymin))
        return ylim
    else:
        ax.set_ylim(ylim)



def cross_section(U, x0, node_mid, nNodes) :
    fig, ax = plt.subplots()
    cross_section_U = U[:,:,node_mid::nNodes]
    cross_section_x0 = x0[node_mid::nNodes,:,:]

    scale = 100
    cross_section_x = cross_section_x0 + scale * np.moveaxis(cross_section_U, (0,1,2), (1, 2, 0))   #from (3, 4, nLayers) to (nLayers, 3, 4)

    plot_cross_section(ax, cross_section_x, plt.cm.plasma, True)
    plot_cross_section(ax, cross_section_x0, plt.cm.Greys, False)
    plot_cross_section(ax, cross_section_x, plt.cm.plasma, False)

    # Add labels, title, and grid
    plt.xlabel('n (mm)')
    plt.ylabel('b (mm)')
    plt.title('Beads defined by particle position')
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling for x and y axes

    plt.show()



def plot_additive(U, qp2elem, nQP, x, Elems, L, Hn, Hb, field, scfplot):
    """
    field can be stt stn stb
    """
    if field == 't':
        projection_index = 0
        label = 't'
    elif field == 'n':
        projection_index = 1
        label = 'n'
    elif field == 'b':
        projection_index = 2
        label = 'b'
    else:
        return 'field type is unknown, should be among t n b'

    uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))

    clr = np.average(uplot[:, projection_index, :], axis=1)

    fig = plt.figure()
    ax = plt.axes(projection='3d', proj_type='ortho')
    ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scfplot), np.ptp(x[:, 2] * scfplot)))

    srf, tri = plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
    colorbar = plt.colorbar(srf, pad=0.15)
    colorbar.set_label(f'{label} [mm]')
















