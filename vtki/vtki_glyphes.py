import vtki
import numpy as np

# Make a grid:
x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                  np.linspace(-5, 5, 20),
                  np.linspace(-5, 5, 5))

grid = vtki.StructuredGrid(x, y, z)

vectors = np.sin(grid.points)**3


# Compute a direction for the vector field
grid.point_arrays['mag'] = np.linalg.norm(vectors, axis=1)
grid.point_arrays['vec'] = vectors

# Make a geometric obhect to use as the glyph
geom = vtki.Arrow() # This could be any dataset

# Perform the glyph
glyphs = grid.glyph(orient='vec', scale='mag', factor=0.8, geom=geom)

# plot using the plotting class
p = vtki.Plotter()
p.add_mesh(glyphs)
p.show()