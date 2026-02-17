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

class Graph_Plotter_With_US_Map:
    """
    Plot a NetworkX graph on top of a US states basemap.

    Two modes:
      1) Geo mode (recommended): pos is lon/lat (EPSG:4326) and we project everything to a US-friendly projection.
      2) Normalized mode: pos is already in your custom normalized x/y space; in that case, you must provide a
         matching transform for the polygons (see notes in __init__).

    Dependencies (recommended):
      - geopandas
      - shapely
      - pyproj

    You must provide a US states polygon dataset (shapefile / geopackage / geojson).
    Example sources: Natural Earth Admin-1, US Census cartographic boundary files.
    """

    def __init__(
        self,
        graph,
        out,
        ax,
        pos,
        us_states_path,
        *,
        pos_is_lonlat=True,
        # Projection to use for plotting (x/y). EPSG:5070 is NAD83 / Conus Albers (great for contiguous US).
        plot_crs="EPSG:5070",
        # CRS of the states file (often EPSG:4326). If None, we assume it’s declared in the file.
        states_src_crs=None,
        # Styling
        state_edgecolor="0.35",
        state_facecolor="none",
        state_linewidth=0.6,
        draw_coastline=False,   # you can ignore this unless you add a coastline layer yourself
        # Node/edge defaults
        node_size=100,
        edge_alpha=0.35,
    ):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.ax = ax
        self.out = out
        self.vmin = float(np.min(out))
        self.vmax = float(np.max(out))
        self.cmap = matplotlib.colormaps["plasma"]

        self.node_size = node_size
        self.edge_alpha = edge_alpha

        # --- load and plot US states polygons ---
        # We do this first so the graph is on top.
        self._plot_crs = plot_crs
        self._pos_is_lonlat = pos_is_lonlat

        self._gdf_states = self._load_states(us_states_path, states_src_crs)
        self._plot_states(self._gdf_states, plot_crs, state_edgecolor, state_facecolor, state_linewidth)
        self.focus_conus(pad=100_000)

        # --- prepare node positions ---
        # If pos is lon/lat, project them to plot_crs so NetworkX draws correctly.
        if pos_is_lonlat:
            self.pos = self._project_pos_lonlat_to_xy(pos, plot_crs)
        else:
            # pos is already x/y in the same coordinate system you want to plot in
            # IMPORTANT: your basemap must be in the same coordinate system too.
            self.pos = pos

        # --- draw base edges lightly (optional) ---
        nx.draw_networkx_edges(self.graph, self.pos, alpha=self.edge_alpha, ax=self.ax)

        # Nice framing:
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_axis_off()

        self.nc = None  # will be set after plot_stationary

    # ---------------------------
    # Public plotting API (mirrors your old class)
    # ---------------------------
    def draw_labels(self, labels=None, font_size=8, x_offset=50000, y_offset=50000):
        """
        In projected meters (EPSG:5070), offsets should be in meters (e.g., 50_000).
        If you use normalized coords, use small offsets (e.g., 0.02).
        """
        if labels is None:
            labels = {node: str(node) for node in self.nodes}

        for node, (x, y) in self.pos.items():
            label = labels.get(node, str(node))

            # Keep your special-cases (adjust if needed)
            if label == "vermont":
                self.ax.text(x - 120000, y - 30000, label, fontsize=font_size, ha="center", va="center")
            elif label in {"rhode island", "new jersey", "north carolina"}:
                self.ax.text(x - 100000, y - 50000, label, fontsize=font_size, ha="center", va="center")
            else:
                self.ax.text(x + x_offset, y + y_offset, label, fontsize=font_size, ha="center", va="center")

    def plot_graph_wieghts(self, weights, ax=None, mapping=None, vmin=None, vmax=None, width=2, zorder=5):
        if ax is None:
            ax = self.ax
        if mapping is None:
            mapping = lambda x: x

        weights = np.asarray(mapping(weights))
        edges = list(self.graph.edges())

        m = min(len(weights), len(edges))
        weights = weights[:m]
        edges = edges[:m]

        if vmin is None:
            vmin = float(weights.min())
        if vmax is None:
            vmax = float(weights.max())

        edge_collection = nx.draw_networkx_edges(
            self.graph,
            self.pos,
            edgelist=edges,
            edge_color=weights,
            edge_cmap=plt.cm.Reds,
            edge_vmin=vmin,
            edge_vmax=vmax,
            width=width,
            ax=ax,
        )
        edge_collection.set_zorder(zorder)

        # Keep consistent framing on the axis used
        gdf_plot = self._gdf_states.to_crs(self._plot_crs)
        xmin, ymin, xmax, ymax = gdf_plot.total_bounds
        pad = getattr(self, "_last_pad", 150_000)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

        return edge_collection

    def plot_stationary(self, signal, ax=None):
        if ax is None:
            ax = self.ax
        im = nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            nodelist=self.nodes,
            node_color=np.asarray(signal),
            node_size=self.node_size,
            cmap=self.cmap,
            ax=ax,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        self.nc = im

    def update_frame(self, frame):
        if self.nc is None:
            raise RuntimeError("Call plot_stationary(...) once before using update_frame(...) for animations.")
        signal = self.out[frame]
        self.nc.set_array(np.array(signal))
        self.ax.set_title(f"node values at: t={frame}")

    # ---------------------------
    # Internals: geopandas + pyproj
    # ---------------------------

    def _load_states(self, us_states_path, states_src_crs):
        import geopandas as gpd

        gdf = gpd.read_file(us_states_path)

        if gdf.crs is None:
            if states_src_crs is None:
                raise ValueError("The states file has no CRS set. Provide states_src_crs='EPSG:4326' (or correct CRS).")
            gdf = gdf.set_crs(states_src_crs)

        # Keep only contiguous US states + DC
        gdf = self._filter_to_conus(gdf)

        return gdf

    def _plot_states(self, gdf, plot_crs, edgecolor, facecolor, linewidth):
        try:
            gdf_plot = gdf.to_crs(plot_crs)
        except Exception as e:
            raise ValueError(f"Failed to reproject states to {plot_crs}. Check CRS of the states file.") from e

        gdf_plot.plot(ax=self.ax, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, zorder=0)

    def _project_pos_lonlat_to_xy(self, pos_lonlat, plot_crs):
        """
        pos_lonlat: dict {node: (lon, lat)} or {node: np.array([lon, lat])}
        returns: dict {node: (x, y)} in plot_crs units (meters for EPSG:5070)
        """
        try:
            from pyproj import Transformer
        except ImportError as e:
            raise ImportError(
                "pyproj is required to project lon/lat node positions. Install with: pip install pyproj"
            ) from e

        # Source is lon/lat WGS84
        transformer = Transformer.from_crs("EPSG:4326", plot_crs, always_xy=True)

        out = {}
        for k, v in pos_lonlat.items():
            lon = float(v[0])
            lat = float(v[1])
            x, y = transformer.transform(lon, lat)
            out[k] = (x, y)
        return out

    def _filter_to_conus(self, gdf):
        """
        Keep only contiguous US states + DC (no AK/HI/territories).
        Works for Natural Earth admin_1_states_provinces_lakes in most versions.
        """
        # First filter to USA if possible
        if "admin" in gdf.columns:
            gdf = gdf[gdf["admin"] == "United States of America"]

        # Try common name columns
        name_col = None
        for c in ["name", "name_en", "name_long", "gn_name"]:
            if c in gdf.columns:
                name_col = c
                break

        if name_col is not None:
            drop = {"Alaska", "Hawaii", "Puerto Rico"}
            gdf = gdf[~gdf[name_col].isin(drop)]

        # If there is a type column, keep only states + DC
        type_col = "type" if "type" in gdf.columns else ("type_en" if "type_en" in gdf.columns else None)
        if type_col is not None:
            gdf = gdf[gdf[type_col].isin(["State", "District"])]

        return gdf

    def focus_conus(self, pad=150_000):
        self._last_pad = pad
        gdf_plot = self._gdf_states.to_crs(self._plot_crs)
        xmin, ymin, xmax, ymax = gdf_plot.total_bounds
        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)

