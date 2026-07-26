import warnings
from typing import IO, Optional, Union

import numpy as np
from numpy.linalg import eigh

from ase import Atoms
from ase.optimize.optimize import Optimizer, UnitCellFilter

# gradient descent optimizer
class GD(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 1.0}

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = '-',
        trajectory: Optional[str] = None,
        append_trajectory: bool = False,
        maxstep: Optional[float] = None,
        master: Optional[bool] = None,
        alpha: Optional[float] = None,
    ):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Scaling factor of step size.
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % self.maxstep)

        self.alpha = alpha
        if self.alpha is None:
            self.alpha = self.defaults['alpha']

        Optimizer.__init__(self, atoms=atoms, restart=restart,
                           logfile=logfile, trajectory=trajectory,
                           master=master, append_trajectory=append_trajectory)

    def initialize(self):
        pass

    def read(self):
        pass

    def step(self, forces=None):
        optimizable = self.optimizable

        if forces is None:
            forces = optimizable.get_forces()

        pos = optimizable.get_positions()
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        optimizable.set_positions(pos + dpos)

        if isinstance(self.atoms, UnitCellFilter):
            self.dump((self.pos0, self.forces0, self.maxstep,
                       self.atoms.orig_cell))
        else:
            self.dump((self.pos0, self.forces0, self.maxstep))

    def prepare_step(self, pos, forces):

        forces = forces.reshape(-1)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dpos = self.alpha * forces
        steplengths = (dpos**2).sum(1)**0.5

        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()

        return dpos, steplengths

    def determine_step(self, dpos, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the direction.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dpos *= scale
        return dpos

