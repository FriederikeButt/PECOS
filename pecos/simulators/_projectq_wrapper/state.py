#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

"""
A simple wrapper for the ProjectQ simulator.

Compatibility checked for: ProjectQ version 0.3.6
"""

from .. import BaseSim
from ...circuits import QuantumCircuit
from projectq import MainEngine
from projectq.ops import All, Measure
from . import bindings
from .logical_sign import find_logical_signs
from .helper import MakeFunc

from qutip import sigmaz, Qobj, tensor,basis, qeye


class ProjectQSim(BaseSim):
    """
    Initializes the stabilizer state.

    Args:
        num_qubits (int): Number of qubits being represented.

    Returns:

    """

    def __init__(self, num_qubits):

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits
        self.eng = MainEngine()

        self.qureg = self.eng.allocate_qureg(num_qubits)
        self.qs = list(self.qureg)
        self.qids = {i: q for i, q in enumerate(self.qs)}
        
        #checks if we need to compute the qutip variables that 
        #are needed for the is_one function
        self.qtipvariables = True

    def cheat(self):
        self.eng.flush()         
        return self.eng.backend.cheat()

    def is_one(self,location):
        
        '''
        Obtains the probability of a qubit in location to be 1, by applying 
        the projector.
        
        First, we obtain the full state |psi> by using the cheat function
        
        then we apply the projector (Id - Z_location)/2 defined in the
        function self. computeQtipVariables()
        
        and we obtain the probability of the qubit being in 1 by doing:
            
            return <psi| (Id - Z_location)/2  |psi>
        '''
        
        #First, we check that we have defined all the Z projectors in qutip
        if self.qtipvariables:
            self.computeQtipVariables()
            self.qtipvariables = False
        
        
        #Now, we obtain the state using the cheat function
        qtipstate = Qobj([self.cheat()[1]])
        qtipstate = qtipstate.trans()
        qtipstate.dims = self.qtipdims
        
        
        #and we obtain the probability of the qubit in location being 1
        #by applying the Z projector (1-Z_location)/2
                
        
        probability = qtipstate.dag()*self.qprojector[location]*qtipstate
        
        return probability
    
    def computeQtipVariables(self):
        '''
        compute all qutip variables needed for the is_one function
        This includes:
            dimensions for the state
            Z projectors for all qubits on the 1 state:
                qprojector[i] = (Id - Z_i)/2
        '''
        
        #in this variable, we store the dimensions of the quantum state
        self.qtipdims = tensor([basis(2,0)]*self.num_qubits).dims
    
        
        #qprojector is a list of all the projectors (1-Z_location)/2 that
        #project the state into the 1 state
        self.qprojector = []
        
        #we store the list of identities
        qeyes = [qeye(2)]*self.num_qubits
        
        identity = tensor(qeyes)
        
        #now we go through all qubits
        for i in range(self.num_qubits):
            
            
            #for each qubit, we need to change the operation in the
            # i-th qubit.
            '''
            Note that in qutip, the qubits are stored in the opposite order. 
            Therefore, to access Z in the qubit i, we need to look at the 
            position N-i. Also, in python positions go from 0 to (N-1), 
            so we need to apply Z in the N-1-i position
            '''
            #with this we obtain a Z operator in the i qubit
            oplist = list(qeyes)            
            oplist[self.num_qubits-i-1] = sigmaz()
            
            Z_i = tensor(oplist)
            
            projector_i = (identity - Z_i)/2
            self.qprojector.append(projector_i)
        
        return
        
    def logical_sign(self, logical_op: QuantumCircuit, allow_float= False) -> int:
        """

        Args:
            logical_op:

        Returns:

        """
        return find_logical_signs(self, logical_op,allow_float)

    def add_gate(self,
                 symbol: str,
                 gate_obj,
                 make_func: bool = True):
        """
        Adds a new gate on the fly to the this Simulator.

        Args:
            symbol:
            gate_obj:
            make_func:

        Returns:

        """

        if symbol in self.gate_dict:
            print('WARNING: Can not add gate as the symbol has already been taken.')
        else:
            if make_func:
                self.gate_dict[symbol] = MakeFunc(gate_obj).func
            else:
                self.gate_dict[symbol] = gate_obj

    def __del__(self):
        self.eng.flush()
        All(Measure) | self.qureg  # Requirement by ProjectQ...

        try:
            self.eng.flush(deallocate_qubits=True)
        except KeyError:
            pass

        # super().__del__()
