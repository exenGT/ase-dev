# fmt: off

import itertools
from math import cos, sin, sqrt
from os.path import basename

import numpy as np

from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from ase.geometry import complete_cell
from ase.gui.colors import ColorWindow
from ase.gui.i18n import ngettext
from ase.gui.render import Render
from ase.gui.repeat import Repeat
from ase.gui.rotate import Rotate
from ase.gui.utils import get_magmoms
from ase.utils import rotate

GREEN = '#74DF00'
PURPLE = '#AC58FA'
BLACKISH = '#151515'


def get_cell_coordinates(cell, shifted=False):
    """Get start and end points of lines segments used to draw cell."""
    nn = []
    for c in range(3):
        v = cell[c]
        d = sqrt(np.dot(v, v))
        if d < 1e-12:
            n = 0
        else:
            n = max(2, int(d / 0.3))
        nn.append(n)
    B1 = np.zeros((2, 2, sum(nn), 3))
    B2 = np.zeros((2, 2, sum(nn), 3))
    n1 = 0
    for c, n in enumerate(nn):
        n2 = n1 + n
        h = 1.0 / (2 * n - 1)
        R = np.arange(n) * (2 * h)

        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            B1[i, j, n1:n2, c] = R
            B1[i, j, n1:n2, (c + 1) % 3] = i
            B1[i, j, n1:n2, (c + 2) % 3] = j
        B2[:, :, n1:n2] = B1[:, :, n1:n2]
        B2[:, :, n1:n2, c] += h
        n1 = n2
    B1.shape = (-1, 3)
    B2.shape = (-1, 3)
    if shifted:
        B1 -= 0.5
        B2 -= 0.5
    return B1, B2


def get_bonds(atoms, bond_cutoff, is_pbc_bonds=True):
    """Find bonds between atoms based on cutoff."""
    from ase.neighborlist import PrimitiveNeighborList

    bond_cutoffs = [0.5 * bond_cutoff for _ in range(len(atoms))]

    nl = PrimitiveNeighborList(
        bond_cutoffs,
        skin=0.0,
        self_interaction=False,
        bothways=False,
    )

    nl.update(atoms.pbc, atoms.get_cell(complete=True), atoms.positions)

    if not is_pbc_bonds:
        for a in range(len(atoms)):
            mask = np.all(nl.displacements[a] == 0, axis=1)
            nl.neighbors[a] = nl.neighbors[a][mask]
            nl.displacements[a] = nl.displacements[a][mask]

    number_of_neighbors = sum(indices.size for indices in nl.neighbors)
    number_of_pbc_neighbors = sum(
        offsets.any(axis=1).sum() for offsets in nl.displacements
    )  # sum up all neighbors that have non-zero supercell offsets
    nbonds = number_of_neighbors + number_of_pbc_neighbors

    bonds = np.empty((nbonds, 5), int)
    if nbonds == 0:
        return bonds

    n1 = 0
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        n2 = n1 + len(indices)
        bonds[n1:n2, 0] = a
        bonds[n1:n2, 1] = indices
        bonds[n1:n2, 2:] = offsets
        n1 = n2

    i = bonds[:n2, 2:].any(1)
    pbcbonds = bonds[:n2][i]
    bonds[n2:, 0] = pbcbonds[:, 1]
    bonds[n2:, 1] = pbcbonds[:, 0]
    bonds[n2:, 2:] = -pbcbonds[:, 2:]

    return bonds


class View:
    def __init__(self, rotations):
        self.colormode = 'jmol'  # The default colors
        self.labels = None
        self.axes = rotate(rotations)
        self.configured = False
        self.frame = None

        self.colors = {
            i: ('#{:02X}{:02X}{:02X}'.format(*(int(x * 255) for x in rgb)))
            for i, rgb in enumerate(jmol_colors)
        }
        # scaling factors for vectors
        self.force_vector_scale = self.config['force_vector_scale']
        self.velocity_vector_scale = self.config['velocity_vector_scale']
        self.drawing_style = self.config.get('drawing_style', 'ball')
        # ADD THIS LINE:
        self.bond_cutoff = self.config.get('bond_cutoff', 1.5)
        self.emphasize_selected_atoms = self.config.get('emphasize_selected_atoms', False)
        self.emphasize_unselected_as_sticks = self.config.get(
            'emphasize_unselected_as_sticks', False)
        self.emphasize_lighten_factor = self.config.get(
            'emphasize_lighten_factor', 0.7)

        # buttons
        self.b1 = 1  # left
        self.b3 = 3  # right
        if self.config['swap_mouse']:
            self.b1 = 3
            self.b3 = 1

    @property
    def atoms(self):
        return self.images[self.frame]

    def set_frame(self, frame=None, focus=False):
        if frame is None:
            frame = self.frame
        assert frame < len(self.images)
        self.frame = frame
        self.set_atoms(self.images[frame])

        fname = self.images.filenames[frame]
        if fname is None:
            header = 'ase.gui'
        else:
            # fname is actually not necessarily the filename but may
            # contain indexing like filename@0
            header = basename(fname)

        images_loaded_text = ngettext(
            'one image loaded',
            '{} images loaded',
            len(self.images)
        ).format(len(self.images))

        self.window.title = f'{header} — {images_loaded_text}'

        if focus:
            self.focus()
        else:
            self.draw()

    def get_bonds(self, atoms):
        # this method exists rather than just using the standalone function
        # so that it can be overridden by external libraries
        try:
            show_pbc = self.window['toggle-show-bonds-pbc']
        except KeyError:
            show_pbc = self.config.get('show_bonds_pbc', True)
        return get_bonds(atoms,
                         self.bond_cutoff,
                         is_pbc_bonds=show_pbc)

    # Add this method to the View class
    def set_bond_cutoff(self, bond_cutoff):
        """Set a new bond cutoff and refresh the view."""
        self.bond_cutoff = bond_cutoff
        self.set_frame()  # This recalculates bonds and redraws

    def show_pbc_bonds(self):
        try:
            return self.window['toggle-show-bonds-pbc']
        except KeyError:
            return self.config.get('show_bonds_pbc', True)

    def segment_inside_cell(self, atoms, scaled_positions, indices):
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if not pbc.any():
            return True

        tol = 0.001
        points = scaled_positions[list(indices)][:, pbc]
        return bool(((points >= -tol) & (points <= 1.0 + tol)).all())

    def pbc_bond_inside_cell(self, atoms, scaled_positions, a, b, offset):
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if not pbc.any():
            return True

        tol = 0.001
        offset = np.asarray(offset, dtype=float)
        mida = 0.5 * (scaled_positions[a] + scaled_positions[b] + offset)
        midb = 0.5 * (scaled_positions[a] + scaled_positions[b] - offset)
        points = np.array([scaled_positions[a], mida,
                           scaled_positions[b], midb])[:, pbc]
        return bool(((points >= -tol) & (points <= 1.0 + tol)).all())

    def get_boundary_bonds(self, atoms, pbc_atoms):
        from ase.neighborlist import PrimitiveNeighborList

        natoms = len(atoms)
        positions = np.vstack([atoms.positions, self.ghosts])
        real_indices = np.arange(len(positions), dtype=int)
        real_indices[natoms:] = self.ghost_indices

        bond_cutoffs = [0.5 * self.bond_cutoff for _ in range(len(positions))]
        nl = PrimitiveNeighborList(
            bond_cutoffs,
            skin=0.0,
            self_interaction=False,
            bothways=False,
        )
        cell = complete_cell(atoms.cell)
        nl.update(np.zeros(3, dtype=bool), cell, positions)

        scaled = np.linalg.solve(cell.T, positions.T).T
        bonds = []
        seen = set()
        for a in range(len(positions)):
            indices, _offsets = nl.get_neighbors(a)
            for b in indices:
                b = int(b)
                if real_indices[a] == real_indices[b]:
                    continue
                if np.linalg.norm(positions[b] - positions[a]) < 1e-12:
                    continue
                if not self.segment_inside_cell(atoms, scaled, (a, b)):
                    continue
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                bonds.append((int(a), b, 0, 0, 0))

        scaled = atoms.get_scaled_positions(wrap=False)
        seen_pbc = {
            (min(a, b), max(a, b), (0, 0, 0))
            for a, b, *_offset in bonds
            if a < natoms and b < natoms
        }
        for a, b, *offset in self.get_bonds(pbc_atoms):
            offset = tuple(int(x) for x in offset)
            if offset == (0, 0, 0):
                continue
            a = int(a)
            b = int(b)
            scaled_offset = np.asarray(offset) * self.images.repeat
            if not self.pbc_bond_inside_cell(atoms, scaled, a, b,
                                             scaled_offset):
                continue
            if a <= b:
                key = (a, b, offset)
            else:
                key = (b, a, tuple(-x for x in offset))
            if key in seen_pbc:
                continue
            seen_pbc.add(key)
            bonds.append((a, b, *offset))

        return np.array(bonds, int).reshape((-1, 5))

    def set_atoms(self, atoms):
        natoms = len(atoms)

        if self.showing_cell():
            B1, B2 = get_cell_coordinates(atoms.cell,
                                          self.config['shift_cell'])
        else:
            B1 = B2 = np.zeros((0, 3))

        self.ghosts = np.zeros((0, 3))
        self.ghost_indices = np.array([], dtype=int)

        if self.config.get('show_boundary_atoms', False) and atoms.pbc.any():
            scaled = atoms.get_scaled_positions(wrap=False)
            indices = []
            offsets = []
            tol = 0.001

            for i in range(natoms):
                boundary_dirs = {}
                for c in range(3):
                    if atoms.pbc[c]:
                        if -tol < scaled[i, c] < tol:
                            boundary_dirs[c] = 1
                        elif 1.0 - tol < scaled[i, c] < 1.0 + tol:
                            boundary_dirs[c] = -1
                axes = list(boundary_dirs.keys())

                if axes:
                    for L in range(1, len(axes) + 1):
                        for subset in itertools.combinations(axes, L):
                            offset = np.zeros(3)
                            for c in subset:
                                offset[c] = boundary_dirs[c]
                            indices.append(i)
                            offsets.append(offset)

            if indices:
                self.ghost_indices = np.array(indices, dtype=int)
                offsets = np.array(offsets)
                self.ghosts = (atoms.positions[self.ghost_indices] +
                               offsets @ atoms.cell)

        nghosts = len(self.ghosts)

        if self.showing_bonds():
            atomscopy = atoms.copy()
            atomscopy.cell *= self.images.repeat[:, np.newaxis]
            if nghosts > 0 and self.show_pbc_bonds():
                bonds = self.get_boundary_bonds(atoms, atomscopy)
            else:
                bonds = self.get_bonds(atomscopy)
        else:
            bonds = np.empty((0, 5), int)

        # X is all atomic coordinates, and starting points of vectors
        # like bonds and cell segments.
        # The reason to have them all in one big list is that we like to
        # eventually rotate/sort it by Z-order when rendering.

        # Also B are the end points of line segments.

        self.X = np.empty((natoms + nghosts + len(B1) + len(bonds), 3))
        self.X_pos = self.X[:natoms]
        self.X_pos[:] = atoms.positions

        if nghosts > 0:
            self.X[natoms:natoms + nghosts] = self.ghosts

        self.X_cell = self.X[natoms + nghosts:natoms + nghosts + len(B1)]
        self.X_bonds = self.X[natoms + nghosts + len(B1):]

        cell = atoms.cell
        ncellparts = len(B1)
        nbonds = len(bonds)

        self.X_cell[:] = np.dot(B1, cell)
        self.B = np.empty((ncellparts + nbonds, 3))
        self.B[:ncellparts] = np.dot(B2, cell)

        if nbonds > 0:
            P = self.X[:natoms + nghosts]
            Af = self.images.repeat[:, np.newaxis] * cell
            a = P[bonds[:, 0]]
            b = P[bonds[:, 1]] + np.dot(bonds[:, 2:], Af) - a
            d = (b**2).sum(1)**0.5
            r = self.get_draw_radii()
            if nghosts > 0:
                r = np.concatenate([r, r[self.ghost_indices]])
            x0 = (r[bonds[:, 0]] / d).reshape((-1, 1))
            x1 = (r[bonds[:, 1]] / d).reshape((-1, 1))
            self.X_bonds[:] = a + b * x0
            b *= 1.0 - x0 - x1
            b[bonds[:, 2:].any(1)] *= 0.5
            self.B[ncellparts:] = self.X_bonds + b

    def showing_bonds(self):
        if self.drawing_style in {'ball-and-stick', 'stick'}:
            return True

        if (self.emphasize_selected_atoms and
                self.emphasize_unselected_as_sticks):
            natoms = len(self.atoms)
            if self.images.selected[:natoms].any():
                return True
        return False

    def get_bond_radius(self):
        return 0.15

    def get_draw_radii(self, atoms=None):
        """Return atom radii for the selected drawing style.
        
        In emphasis mode with selected atoms:
        - selected atoms keep the selected drawing style
        - background atoms optionally use bond radius (stick style)
        """
        if atoms is None:
            atoms = self.atoms

        if self.drawing_style == 'stick':
            radii = np.full(len(atoms), self.get_bond_radius())
        else:
            radii = self.get_covalent_radii(atoms)
        
        if (self.emphasize_selected_atoms and
                self.emphasize_unselected_as_sticks):
            natoms = len(atoms)
            selected = self.images.selected[:natoms]
            if selected.any():
                radii = radii.copy()
                radii[~selected] = self.get_bond_radius()

        return radii

    def showing_cell(self):
        return self.window['toggle-show-unit-cell']

    def toggle_show_unit_cell(self, key=None):
        self.set_frame()

    def update_labels(self):
        index = self.window['show-labels']
        if index == 0:
            self.labels = None
        elif index == 1:
            self.labels = list(range(len(self.atoms)))
        elif index == 2:
            self.labels = list(get_magmoms(self.atoms))
        elif index == 4:
            Q = self.atoms.get_initial_charges()
            self.labels = [f'{q:.4g}' for q in Q]
        else:
            self.labels = self.atoms.get_chemical_symbols()

    def show_labels(self):
        self.update_labels()
        self.draw()

    def toggle_show_axes(self, key=None):
        self.draw()

    def toggle_show_bonds(self, key=None):
        self.drawing_style = ('ball-and-stick'
                              if self.window['toggle-show-bonds'] else 'ball')
        self.config['drawing_style'] = self.drawing_style
        self.update_drawing_style_menu()
        self.set_frame()

    def update_drawing_style_menu(self):
        try:
            self.window['set-drawing-style'] = [
                'ball', 'ball-and-stick', 'stick'].index(self.drawing_style)
        except KeyError:
            pass
        try:
            self.window['toggle-show-bonds'] = self.showing_bonds()
        except KeyError:
            pass

    def set_drawing_style(self, key=None):
        styles = ['ball', 'ball-and-stick', 'stick']
        self.drawing_style = styles[self.window['set-drawing-style']]
        self.config['drawing_style'] = self.drawing_style
        self.update_drawing_style_menu()
        self.set_frame()

    def toggle_emphasize_selected_atoms(self, key=None):
        """Toggle the emphasis mode for selected atoms."""
        self.emphasize_selected_atoms = not self.emphasize_selected_atoms
        self.config['emphasize_selected_atoms'] = self.emphasize_selected_atoms
        self.config['emphasize_unselected_as_sticks'] = (
            self.emphasize_unselected_as_sticks)
        self.config['emphasize_lighten_factor'] = self.emphasize_lighten_factor
        self.update_drawing_style_menu()
        self.set_frame()

    def toggle_show_bonds_pbc(self, key=None):
        self.set_frame()

    def toggle_show_boundary_atoms(self, key=None):
        self.config['show_boundary_atoms'] = not self.config.get(
            'show_boundary_atoms', False)
        self.set_frame()

    def toggle_show_velocities(self, key=None):
        self.draw()

    def get_forces(self):
        if self.atoms.calc is not None:
            try:
                return self.atoms.get_forces()
            except PropertyNotImplementedError:
                pass
        return np.zeros((len(self.atoms), 3))

    def toggle_show_forces(self, key=None):
        self.draw()

    def hide_selected(self):
        self.images.visible[self.images.selected] = False
        self.draw()

    def show_selected(self):
        self.images.visible[self.images.selected] = True
        self.draw()

    def repeat_window(self, key=None):
        return Repeat(self)

    def rotate_window(self):
        return Rotate(self)

    def colors_window(self, key=None):
        win = ColorWindow(self)
        self.register_vulnerable(win)
        return win

    def focus(self, x=None):
        cell = (self.window['toggle-show-unit-cell'] and
                self.images[0].cell.any())
        if (len(self.atoms) == 0 and not cell):
            self.scale = 20.0
            self.center = np.zeros(3)
            self.draw()
            return

        # Get the min and max point of the projected atom positions
        # including the covalent_radii used for drawing the atoms
        P = np.dot(self.X, self.axes)
        n = len(self.atoms)
        covalent_radii = self.get_draw_radii()
        P[:n] -= covalent_radii[:, None]
        P1 = P.min(0)
        P[:n] += 2 * covalent_radii[:, None]
        P2 = P.max(0)
        self.center = np.dot(self.axes, (P1 + P2) / 2)
        self.center += self.atoms.get_celldisp().reshape((3,)) / 2
        # Add 30% of whitespace on each side of the atoms
        S = 1.3 * (P2 - P1)
        w, h = self.window.size
        if S[0] * h < S[1] * w:
            self.scale = h / S[1]
        elif S[0] > 0.0001:
            self.scale = w / S[0]
        else:
            self.scale = 1.0
        self.draw()

    def reset_view(self, menuitem):
        self.axes = rotate('0.0x,0.0y,0.0z')
        self.set_frame()
        self.focus(self)

    def set_view(self, key):
        if key == 'Z':
            self.axes = rotate('0.0x,0.0y,0.0z')
        elif key == 'X':
            self.axes = rotate('-90.0x,-90.0y,0.0z')
        elif key == 'Y':
            self.axes = rotate('90.0x,0.0y,90.0z')
        elif key == 'Alt+Z':
            self.axes = rotate('180.0x,0.0y,90.0z')
        elif key == 'Alt+X':
            self.axes = rotate('0.0x,90.0y,0.0z')
        elif key == 'Alt+Y':
            self.axes = rotate('-90.0x,0.0y,0.0z')
        else:
            if key == '3':
                i, j = 0, 1
            elif key == '1':
                i, j = 1, 2
            elif key == '2':
                i, j = 2, 0
            elif key == 'Alt+3':
                i, j = 1, 0
            elif key == 'Alt+1':
                i, j = 2, 1
            elif key == 'Alt+2':
                i, j = 0, 2

            A = complete_cell(self.atoms.cell)
            x1 = A[i]
            x2 = A[j]

            norm = np.linalg.norm

            x1 = x1 / norm(x1)
            x2 = x2 - x1 * np.dot(x1, x2)
            x2 /= norm(x2)
            x3 = np.cross(x1, x2)

            self.axes = np.array([x1, x2, x3]).T

        self.set_frame()

    def get_colors(self, rgb=False):
        if rgb:
            return [tuple(int(_rgb[i:i + 2], 16) / 255 for i in range(1, 7, 2))
                    for _rgb in self.get_colors()]

        if self.colormode == 'jmol':
            return [self.colors.get(Z, BLACKISH) for Z in self.atoms.numbers]

        if self.colormode == 'neighbors':
            return [self.colors.get(Z, BLACKISH)
                    for Z in self.get_color_scalars()]

        colorscale, cmin, cmax = self.colormode_data
        N = len(colorscale)
        colorswhite = colorscale + ['#ffffff']
        if cmin == cmax:
            indices = [N // 2] * len(self.atoms)
        else:
            scalars = np.ma.array(self.get_color_scalars())
            indices = np.clip(((scalars - cmin) / (cmax - cmin) * N +
                               0.5).astype(int),
                              0, N - 1).filled(N)
        return [colorswhite[i] for i in indices]

    def lighten_hex_color(self, color, factor):
        rgb = [int(color[i:i + 2], 16) / 255 for i in range(1, 7, 2)]
        rgb = [c + factor * (1.0 - c) for c in rgb]
        return '#{:02X}{:02X}{:02X}'.format(
            *(round(255 * c) for c in rgb))

    def should_lighten_atom(self, index):
        if not self.emphasize_selected_atoms:
            return False
        selected = self.images.selected[:len(self.atoms)]
        return selected.any() and not selected[index]

    def get_emphasized_color(self, color, index):
        if self.should_lighten_atom(index):
            return self.lighten_hex_color(color, self.emphasize_lighten_factor)
        return color

    def get_emphasized_colors(self, colors):
        return [self.get_emphasized_color(color, index)
                for index, color in enumerate(colors)]

    def get_color_scalars(self, frame=None):
        if self.colormode == 'tag':
            return self.atoms.get_tags()
        if self.colormode == 'force':
            f = (self.get_forces()**2).sum(1)**0.5
            return f * self.images.get_dynamic(self.atoms)
        elif self.colormode == 'velocity':
            return (self.atoms.get_velocities()**2).sum(1)**0.5
        elif self.colormode == 'initial charge':
            return self.atoms.get_initial_charges()
        elif self.colormode == 'magmom':
            return get_magmoms(self.atoms)
        elif self.colormode == 'neighbors':
            from ase.neighborlist import NeighborList
            n = len(self.atoms)
            nl = NeighborList(self.get_covalent_radii(self.atoms) * 1.5,
                              skin=0, self_interaction=False, bothways=True)
            nl.update(self.atoms)
            return [len(nl.get_neighbors(i)[0]) for i in range(n)]
        else:
            scalars = np.array(self.atoms.get_array(self.colormode),
                               dtype=float)
            return np.ma.array(scalars, mask=np.isnan(scalars))

    def get_covalent_radii(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        return self.images.get_radii(atoms)

    def draw(self, status=True):
        self.window.clear()
        axes = self.scale * self.axes * (1, -1, 1)
        offset = np.dot(self.center, axes)
        offset[:2] -= 0.5 * self.window.size
        X = np.dot(self.X, axes) - offset
        n = len(self.atoms)
        nghosts = len(self.ghost_indices) if hasattr(self, 'ghost_indices') else 0

        # The indices enumerate drawable objects in z order:
        self.indices = X[:, 2].argsort()
        r = self.get_draw_radii() * self.scale

        if nghosts > 0:
            r = np.concatenate([r, r[self.ghost_indices]])

        P = self.P = X[:n + nghosts, :2]
        A = (P - r[:, None]).round().astype(int)
        X1 = X[n + nghosts:, :2].round().astype(int)
        X2 = (np.dot(self.B, axes) - offset).round().astype(int)
        disp = (np.dot(self.atoms.get_celldisp().reshape((3,)),
                       axes)).round().astype(int)
        d = (2 * r).round().astype(int)

        vector_arrays = []
        if self.window['toggle-show-velocities']:
            # Scale ugly?
            v = self.atoms.get_velocities()
            if v is not None:
                vector_arrays.append(v * 10.0 * self.velocity_vector_scale)
        if self.window['toggle-show-forces']:
            f = self.get_forces()
            vector_arrays.append(f * self.force_vector_scale)

        for array in vector_arrays:
            array[:] = np.dot(array, axes) + X[:n]

        colors = self.get_emphasized_colors(self.get_colors())
        circle = self.window.circle
        arc = self.window.arc
        line = self.window.line
        constrained = ~self.images.get_dynamic(self.atoms)

        selected = self.images.selected
        visible = self.images.visible
        ncell = len(self.X_cell)
        bond_linewidth = self.scale * 2 * self.get_bond_radius()

        self.update_labels()

        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            movecolor = GREEN
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            movecolor = PURPLE

        for a in self.indices:
            if a < n + nghosts:
                if a < n:
                    real_a = a
                else:
                    real_a = self.ghost_indices[a - n]

                ra = d[a]
                if visible[real_a]:
                    try:
                        kinds = self.atoms.arrays['spacegroup_kinds']
                        site_occ = self.atoms.info['occupancy'][str(kinds[real_a])]
                        # first an empty circle if a site is not fully occupied
                        if (np.sum([v for v in site_occ.values()])) < 1.0:
                            fill = '#ffffff'
                            circle(fill, selected[real_a],
                                   A[a, 0], A[a, 1],
                                   A[a, 0] + ra, A[a, 1] + ra)
                        start = 0
                        # start with the dominant species
                        for sym, occ in sorted(site_occ.items(),
                                               key=lambda x: x[1],
                                               reverse=True):
                            if np.round(occ, decimals=4) == 1.0:
                                circle(colors[real_a], selected[real_a],
                                       A[a, 0], A[a, 1],
                                       A[a, 0] + ra, A[a, 1] + ra)
                            else:
                                # jmol colors for the moment
                                extent = 360. * occ
                                arc(self.get_emphasized_color(
                                    self.colors[atomic_numbers[sym]], real_a),
                                    selected[real_a],
                                    start, extent,
                                    A[a, 0], A[a, 1],
                                    A[a, 0] + ra, A[a, 1] + ra)
                                start += extent
                    except KeyError:
                        # legacy behavior
                        # Draw the atoms
                        if (self.moving and real_a < len(self.move_atoms_mask)
                                and self.move_atoms_mask[real_a]):
                            circle(movecolor, False,
                                   A[a, 0] - 4, A[a, 1] - 4,
                                   A[a, 0] + ra + 4, A[a, 1] + ra + 4)

                        circle(colors[real_a], selected[real_a],
                               A[a, 0], A[a, 1], A[a, 0] + ra, A[a, 1] + ra)

                    # Draw labels on the atoms
                    if self.labels is not None:
                        self.window.text(A[a, 0] + ra / 2,
                                         A[a, 1] + ra / 2,
                                         str(self.labels[real_a]))

                    # Draw cross on constrained atoms
                    if constrained[real_a]:
                        R1 = int(0.14644 * ra)
                        R2 = int(0.85355 * ra)
                        line((A[a, 0] + R1, A[a, 1] + R1,
                              A[a, 0] + R2, A[a, 1] + R2))
                        line((A[a, 0] + R2, A[a, 1] + R1,
                              A[a, 0] + R1, A[a, 1] + R2))

                    # Draw velocities and/or forces
                    if a < n:
                        for v in vector_arrays:
                            assert not np.isnan(v).any()
                            self.arrow((X[a, 0], X[a, 1], v[a, 0], v[a, 1]),
                                       width=2)
            else:
                # Draw unit cell and/or bonds:
                a -= (n + nghosts)
                if a < ncell:
                    line((X1[a, 0] + disp[0], X1[a, 1] + disp[1],
                          X2[a, 0] + disp[0], X2[a, 1] + disp[1]))
                else:
                    line((X1[a, 0], X1[a, 1],
                          X2[a, 0], X2[a, 1]),
                         width=bond_linewidth)

        if self.window['toggle-show-axes']:
            self.draw_axes()

        if len(self.images) > 1:
            self.draw_frame_number()

        self.window.update()

        if status:
            self.status(self.atoms)

    def arrow(self, coords, width):
        line = self.window.line
        begin = np.array((coords[0], coords[1]))
        end = np.array((coords[2], coords[3]))
        line(coords, width)

        vec = end - begin
        length = np.sqrt((vec[:2]**2).sum())
        length = min(length, 0.3 * self.scale)

        angle = np.arctan2(end[1] - begin[1], end[0] - begin[0]) + np.pi
        x1 = (end[0] + length * np.cos(angle - 0.3)).round().astype(int)
        y1 = (end[1] + length * np.sin(angle - 0.3)).round().astype(int)
        x2 = (end[0] + length * np.cos(angle + 0.3)).round().astype(int)
        y2 = (end[1] + length * np.sin(angle + 0.3)).round().astype(int)
        line((x1, y1, end[0], end[1]), width)
        line((x2, y2, end[0], end[1]), width)

    def draw_axes(self):
        axes_length = 15

        rgb = ['red', 'green', 'blue']

        for i in self.axes[:, 2].argsort():
            a = 20
            b = self.window.size[1] - 20
            c = int(self.axes[i][0] * axes_length + a)
            d = int(-self.axes[i][1] * axes_length + b)
            self.window.line((a, b, c, d))
            self.window.text(c, d, 'XYZ'[i], color=rgb[i])

    def draw_frame_number(self):
        x, y = self.window.size
        self.window.text(x, y, '{}'.format(self.frame),
                         anchor='SE')

    def release(self, event):
        if event.button in [4, 5]:
            self.scroll_event(event)
            return

        if event.button != self.b1:
            return

        selected = self.images.selected
        selected_ordered = self.images.selected_ordered
        natoms = len(self.atoms)
        nghosts = len(self.ghost_indices) if hasattr(self, 'ghost_indices') else 0

        def real_atom_index(index):
            if index < natoms:
                return index
            if index < natoms + nghosts:
                return self.ghost_indices[index - natoms]
            return None

        if event.time < self.t0 + 200:  # 200 ms
            d = self.P - self.xy
            r = self.get_draw_radii()
            if nghosts > 0:
                r = np.concatenate([r, r[self.ghost_indices]])
            hit = np.less((d**2).sum(1), (self.scale * r)**2)
            for a in self.indices[::-1]:
                real_a = real_atom_index(a)
                if real_a is not None and hit[a]:
                    if event.modifier == 'ctrl':
                        selected[real_a] = not selected[real_a]
                        if selected[real_a]:
                            selected_ordered += [real_a]
                        elif len(selected_ordered) > 0:
                            if selected_ordered[-1] == real_a:
                                selected_ordered = selected_ordered[:-1]
                            else:
                                selected_ordered = []
                    else:
                        selected[:] = False
                        selected[real_a] = True
                        selected_ordered = [real_a]
                    break
            else:
                selected[:] = False
                selected_ordered = []
            self.draw()
        else:
            A = (event.x, event.y)
            C1 = np.minimum(A, self.xy)
            C2 = np.maximum(A, self.xy)
            hit = np.logical_and(self.P > C1, self.P < C2)
            drawable_indices = np.compress(hit.prod(1), np.arange(len(hit)))
            indices = []
            for index in drawable_indices:
                real_index = real_atom_index(index)
                if real_index is not None:
                    indices.append(real_index)
            indices = np.unique(indices).astype(int)
            if event.modifier != 'ctrl':
                selected[:] = False
            selected[indices] = True
            if (len(indices) == 1 and
                    indices[0] not in self.images.selected_ordered):
                selected_ordered += [indices[0]]
            elif len(indices) > 1:
                selected_ordered = []
            self.draw()

        # XXX check bounds
        indices = np.arange(natoms)[self.images.selected[:natoms]]
        if len(indices) != len(selected_ordered):
            selected_ordered = []
        self.images.selected_ordered = selected_ordered

    def press(self, event):
        self.button = event.button
        self.xy = (event.x, event.y)
        self.t0 = event.time
        self.axes0 = self.axes
        self.center0 = self.center

    def move(self, event):
        x = event.x
        y = event.y
        x0, y0 = self.xy
        if self.button == self.b1:
            x0 = int(round(x0))
            y0 = int(round(y0))
            self.draw()
            self.window.canvas.create_rectangle((x, y, x0, y0))
            return

        if event.modifier == 'shift':
            self.center = (self.center0 -
                           np.dot(self.axes, (x - x0, y0 - y, 0)) / self.scale)
        else:
            # Snap mode: the a-b angle and t should multipla of 15 degrees ???
            a = x - x0
            b = y0 - y
            t = sqrt(a * a + b * b)
            if t > 0:
                a /= t
                b /= t
            else:
                a = 1.0
                b = 0.0
            c = cos(0.01 * t)
            s = -sin(0.01 * t)
            rotation = np.array([(c * a * a + b * b, (c - 1) * b * a, s * a),
                                 ((c - 1) * a * b, c * b * b + a * a, s * b),
                                 (-s * a, -s * b, c)])
            self.axes = np.dot(self.axes0, rotation)
            if len(self.atoms) > 0:
                com = self.X_pos.mean(0)
            else:
                com = self.atoms.cell.mean(0)
            self.center = com - np.dot(com - self.center0,
                                       np.dot(self.axes0, self.axes.T))
        self.draw(status=False)

    def render_window(self):
        return Render(self)

    def resize(self, event):
        w, h = self.window.size
        self.scale *= (event.width * event.height / (w * h))**0.5
        self.window.size[:] = [event.width, event.height]
        self.draw()
