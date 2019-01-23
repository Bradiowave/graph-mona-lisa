#!/usr/bin/env python

#======== graph.py ============
"""
Simple graph implementation compatible with BokehGraph class.
"""
class Vertex:
    def __init__(self, label, component=-1):
        self.label = str(label)
        self.component = component

    def __repr__(self):
        return 'Vertex: ' + self.label

class Graph:
    """Trying to make this Graph class work..."""
    def __init__(self):
        self.vertices = {}
        self.components = 0

    def add_vertex(self, vertex, edges=()):
        self.vertices[vertex] = set(edges) 

    def add_edge(self, start, end, bidirectional=True):
        self.vertices[start].add(end)
        if bidirectional:
            self.vertices[end].add(start)

    def dfs(self, start, target=None):
        stack = []
        stack.append(start)
        visited = set(stack)

        while stack:
            current = stack.pop()
            visited.add(current)
            if current == target:
                break
            stack.extend(self.vertices[current] - visited)

        return visited

    def graph_rec(self, start, visited=None, target=None):
        visited = visited or set()
        visited.add(start)
        for vertex in self.vertices[start]:
            self.graph_rec(vertex, visited=visited)
        return visited

    def find_components(self):
        visited = set()
        current_component = 0

        for vertex in self.vertices:
            if vertex not in visited:
                reachable = self.dfs(vertex)
                for other_vertex in reachable:
                    other_vertex.component = current_component
                current_component += 1
                visited.update(reachable)
        self.components = current_component

#======== graph.py END ========


#======== draw.py ============
"""
General drawing methods for graphs using Bokeh.
"""
import cv2
from math import ceil, floor, sqrt
from random import choice, random
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import (GraphRenderer, StaticLayoutProvider, Circle, LabelSet,
                          ColumnDataSource)


class BokehGraph:
    """Class that takes a graph and exposes drawing methods."""
    def __init__(self, graph, title='Graph', width=100, height=100,
                 show_axis=False, show_grid=False, circle_size=35,
                 draw_components=False):
        if not graph.vertices:
            raise Exception('Graph should contain vertices!')
        self.graph = graph
        self.width = width
        self.height = height

        self.reasonable = ( ( (self.width)**2 + (self.height)**2 ) ** (1/2.0) ) / ((len(self.graph.vertices)+1) * 2) ** (1/1.7)

        self.circle_size = ( ( (self.width)**2 + (self.height)**2 ) ** (1/2.0) ) / ((len(self.graph.vertices)+1) * 2) ** (1/4)
        self.font_size = str(int(self.circle_size/2.5)) + 'pt'
        self.pos = {}  # dict to map vertices to x, y positions
        # Set up plot, the canvas/space to draw on
        self.plot = figure(title=title, x_range=(0, width), y_range=(0, height))
        self.plot.axis.visible = show_axis
        self.plot.grid.visible = show_grid
        self._setup_graph_renderer(circle_size, draw_components)
        self._setup_labels()

    def _setup_graph_renderer(self, circle_size, draw_components):
        # The renderer will have the actual logic for drawing
        graph_renderer = GraphRenderer()
        # Saving vertices in an arbitrary but persistent order
        self.vertex_list = list(self.graph.vertices.keys())

        # Add the vertex data as instructions for drawing nodes
        graph_renderer.node_renderer.data_source.add(
            [vertex.label for vertex in self.vertex_list], 'index')
        colors = (self._image_colors() if draw_components
                  else self._get_random_colors())
        graph_renderer.node_renderer.data_source.add(colors, 'color')
        # And circles
        graph_renderer.node_renderer.glyph = Circle(size=self.circle_size,
                                                    fill_color='color')

        # Add the edge [start, end] indices as instructions for drawing edges
        graph_renderer.edge_renderer.data_source.data = self._get_edge_indexes()
        # self.randomize()  # Randomize vertex coordinates, and set as layout
        graph_renderer.layout_provider = StaticLayoutProvider(
            graph_layout=self.pos)
        # Attach the prepared renderer to the plot so it can be shown
        self.plot.renderers.append(graph_renderer)

    def _get_random_colors(self, num_colors=None):
        colors = []
        num_colors = num_colors or len(self.graph.vertices)
        for _ in range(num_colors):
            color = '#'+''.join([choice('0123456789ABCDEF') for j in range(6)])
            colors.append(color)
        return colors

    def _get_edge_indexes(self):
        start_indices = []
        end_indices = []
        checked = set()

        for vertex, edges in self.graph.vertices.items():
            if vertex not in checked:
                for destination in edges:
                    start_indices.append(vertex.label)
                    end_indices.append(destination.label)
                checked.add(vertex)

        return dict(start=start_indices, end=end_indices)

    def _setup_labels(self):
        label_data = {'x': [], 'y': [], 'names': []}
        for vertex_label, (x_pos, y_pos) in self.pos.items():
            label_data['x'].append(x_pos)
            label_data['y'].append(y_pos)
            label_data['names'].append(vertex_label)
        label_source = ColumnDataSource(label_data)
        labels = LabelSet(x='x', y='y', text='names', level='glyph', text_color='white',
                          text_font_size=self.font_size, text_align='center', text_baseline='middle',
                          source=label_source, render_mode='canvas')
        # self.plot.add_layout(labels)

    def show(self, output_path='./graph.html'):
        """Render the graph to a file on disk and open with default browser."""
        output_file(output_path)
        show(self.plot)

    def randomize(self):
        """Randomize vertex positions."""
        for vertex in self.vertex_list:
            # TODO make bounds and random draws less hacky

            while True:
                randx = 1 + random() * (self.width - 2)
                randy = 1 + random() * (self.height - 2)
                touchy = False

                for vert in self.pos:
                    x1 = self.pos[vert][0]
                    y1 = self.pos[vert][1]
                    distance_to_vert = ( ( (randx - x1)**2 ) + ( (randy - y1)**2 ) ) ** (1/2.0)
                    if distance_to_vert < self.reasonable : # or ( randx < self.circle_size or randx > self.width-self.circle_size ) or ( randy < self.circle_size or randy > self.height+self.circle_size )
                        # print ('touchy')
                        touchy = True
                        break

                if not touchy:
                    break

            self.pos[vertex.label] = (randx , randy)

    def _get_connected_component_colors(self):
        """Return same-colors for vertices in connected components."""
        self.graph.find_components()
        component_colors = self._get_random_colors(self.graph.components)
        vertex_colors = []
        for vertex in self.vertex_list:
            vertex_colors.append(component_colors[vertex.component])
        return vertex_colors

    def _image_colors(self):
        self.randomize()
        im = cv2.imread('monalisa.png')

        def convert_pos_to_color(position):
            bgr = im[int(position[0])][int(position[1])]
            rgb = (bgr[2], bgr[1], bgr[0])
            hex_color = '#%02x%02x%02x' % rgb
            return hex_color

        image_colors = self.pos.copy()

        for vertex in image_colors:
            image_colors[vertex] = convert_pos_to_color(image_colors[vertex])

        colors = list(image_colors.values())
        return colors

#======== draw.py END ========




"""
Demonstration of Graph and BokehGraph functionality.
"""

from random import sample
from sys import argv

def main(num_vertices=1000, num_edges=100, draw_components=True):
    """Build and show random graph."""
    graph = Graph()
    # Add appropriate number of vertices
    for num in range(num_vertices):
        graph.add_vertex(Vertex(label=str(num)))

    # Add random edges between vertices
    for _ in range(num_edges):
        vertices = sample(graph.vertices.keys(), 2)
        # TODO check if edge already exists
        graph.add_edge(vertices[0], vertices[1])

    bokeh_graph = BokehGraph(graph, draw_components=draw_components)
    bokeh_graph.show()


if __name__ == '__main__':
    if len(argv) == 4:
        NUM_VERTICES = int(argv[1])
        NUM_EDGES = int(argv[2])
        DRAW_COMPONENTS = bool(int(argv[3]))
        main(NUM_VERTICES, NUM_EDGES, DRAW_COMPONENTS)
    else:
        main()
