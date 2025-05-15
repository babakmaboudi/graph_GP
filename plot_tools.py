import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

class Plotter:
    """
    Class for plotting a sample in a plt axis handler.
    """

    def plot_sample(self, u, ax):
        """
        Plot sample in a plt axis handler

        Parameters:
        -----------
        u: displacement (np.array)
        ax: Matplotlib axis object

        Returns:
        --------
        """
        nbin = 64 # number of intensity colors
        cmap = matplotlib.colormaps['plasma'] # choice of color palette
        colors = cmap(np.linspace(0, 1, nbin)) # color codes in RGB

        dist = np.max(u) - np.min(u) # normalizing value to make a double into an integer
        for i in range(self.x.shape[0]-1):
            color_id = int( (u[i]-np.min(u) )/dist * (nbin-1) ) # transforimg a double into an integer between 0 and nbin
            x = self.x[i:i+2]
            y = self.y[i:i+2]
            ax.plot( x, y , color=colors[color_id], linewidth=3 )
        ax.set_aspect('equal')

    def plot_sample_displacement(self, u, ax):
        """
        Plot graph samples as normal displacements (currently has a bug)?

        Parameters:
        -----------
        u: displacement (np.array)
        ax: Matplotlib axis object

        Returns:
        --------
        """
        nbin = 64
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, nbin))

        points = (1+0.05*u)*np.exp( 1j*self.theta )
        xx = np.real(points)
        yy = np.imag(points)


        dist = np.max(u) - np.min(u)
        for i in range(self.x.shape[0]-1):
            color_id = int( (u[i]-np.min(u) )/dist * (nbin-1) )
            x = xx[i:i+2] 
            y = yy[i:i+2]
            ax.plot( x, y, color='blue', linewidth=3 )
        #ax.plot(self.x,self.y)
        #ax.scatter(self.x,self.y, c=u, '.')
        ax.set_aspect('equal')


class Graph_Plotter():
    """
    Handles plotting of graph-based processes.
    """
    def __init__(self, graph, out, ax, pos=None):
        """
        Initializes the GraphPlotter.

        Parameters:
        -----------
        graph: NetworkX graph instance
        out: Simulation output
        ax: Matplotlib axis object
        """
        self.out = out
        self.vmin = np.min(self.out)
        self.vmax = np.max(self.out)
        
        self.graph = graph
        self.nodes = graph.nodes()
        self.ax = ax
        #self.pos = nx.random_layout(self.graph)
        if(pos is None):
            self.pos = nx.spring_layout(self.graph)
        else:
            self.pos = pos
        self.cmap = matplotlib.colormaps['plasma']

        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.2, ax=self.ax)


    def draw_labels(self, labels=None, font_size=8, x_offset=0.02, y_offset=0.02):
        """
        Draw labels for nodes with an offset from the node positions.

        Parameters:
        -----------
        labels: dict of labels, default None
            The labels for each node. If None, node names are used as labels.
        font_size: int, default 8
            The font size of the labels.
        x_offset: float, default 0.02
            The horizontal offset for the label position (positive to move right, negative to move left).
        y_offset: float, default 0.02
            The vertical offset for the label position (positive to move up, negative to move down).

        Returns:
        --------
        None
        """
        if labels is None:
            # Default: label each node with its name
            labels = {node: str(node) for node in self.nodes}

        # Manually adjust the positions and add the labels with offset
        for node, (x, y) in self.pos.items():
            label = labels[node]
            # Offset the labels using data units instead of axes
            if label=="vermont":
                self.ax.text(x -0.05, y -0.01, label, fontsize=font_size, ha='center', va='center')
            elif label=="rhode island":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            elif label=="new jersey":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            elif label=="north carolina":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            else:
                self.ax.text(x + x_offset, y + y_offset, label, fontsize=font_size, ha='center', va='center')


    def plot_graph_wieghts(self, weights, ax, mapping=None, vmin=None, vmax=None):
        if(mapping is None):
            mapping = lambda x: x
        weights = mapping(weights)
        cmap = plt.cm.viridis
        #nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.nodes, node_color=signal, node_size=100, cmap=self.cmap, ax=ax)
        if(vmin is None):
            edge_collection = nx.draw_networkx_edges( self.graph, self.pos, edgelist=self.graph.edges(data=True), edge_color=weights[:110], edge_cmap=plt.cm.Reds, edge_vmin=weights.min(), edge_vmax=weights.max(), width=2, ax=ax)
        else:
                        edge_collection = nx.draw_networkx_edges( self.graph, self.pos, edgelist=self.graph.edges(data=True), edge_color=weights[:110], edge_cmap=plt.cm.Reds, edge_vmin=vmin, edge_vmax=vmax, width=2, ax=ax)
        ax.set_title('graph weights')
        plt.colorbar(edge_collection, ax=ax)


    def plot_stationary(self, signal):
        """
        Plots stationary signals on graph nodes.

        Parameters:
        -----------
        signal: np.array

        Returns:
        --------
        """
        im = self.nc = nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.nodes, node_color=signal, node_size=100, cmap=self.cmap, ax=self.ax, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar(im, ax=self.ax)


    def update_frame(self,frame):
        """
        Updates node colors for animation frames.

        Parameters:
        -----------
        frame: animation frame

        Returns:
        --------
        """
        signal = self.out[frame]
        self.nc.set_array(np.array(signal))
        self.ax.set_title('node values at: t={}'.format(frame))
