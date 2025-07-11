# distutils: language = c++
#cython: language_level=3, boundscheck=False, nonecheck=False, wraparound=False

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

cimport cysparsesim_header as s
from cysparsesim_header cimport int_num

from .src.logical_sign import find_logical_signs


cdef dict gate_dict = {
                'I': State.I,
                'X': State.X,
                'Y': State.Y,
                'Z': State.Z,
                
                'II': State.II,
                'CNOT': State.cnot,
                'CZ': State.cz,
                'CY': State.cy,
                'SWAP': State.swap,
                'G': State.g2,
                
                'H': State.hadamard,
                'H1': State.hadamard,
                'H2': State.H2,
                'H3': State.H3,
                'H4': State.H4,
                'H5': State.H5,
                'H6': State.H6,
                
                'H+z+x': State.hadamard, 
                'H-z-x': State.H2, 
                'H+y-z': State.H3,
                'H-y-z': State.H4,
                'H-x+y': State.H5, 
                'H-x-y': State.H6,
                
                'F1': State.F1,
                'F2': State.F2,
                'F3': State.F3,
                'F4': State.F4,
                
                'F1d': State.F1d,
                'F2d': State.F2d,
                'F3d': State.F3d,
                'F4d': State.F4d,
                
                'R': State.R,
                'Rd': State.Rd,
                'Q': State.Q,
                'Qd': State.Qd,
                'S': State.S,
                'Sd': State.Sd,

                'measure X': State.measureX,
                'measure Y': State.measureY,                
                'measure Z': State.measure,
                'force output': State.force_output,
                
                'init |0>': State.initzero,
                'init |1>': State.initone,
                'init |+>': State.initplus,
                'init |->': State.initminus,
                'init |+i>': State.initplusi,
                'init |-i>': State.initminusi,
                
            }

cdef class State:
    
    cdef s.State* _c_state
    
    cdef public:
        int_num num_qubits
        int reserve_buckets
        dict gate_dict
    
    def __cinit__(self, int_num num_qubits, int reserve_buckets=0):
        self._c_state = new s.State(num_qubits, reserve_buckets)
        
        self.num_qubits = self._c_state.num_qubits
        self.reserve_buckets = self._c_state.reserve_buckets
        self.gate_dict = gate_dict
    
    cdef void hadamard(self, int_num qubit):
        self._c_state.hadamard(qubit)
        
    cdef void H2(self, int_num qubit):
        self._c_state.H2(qubit)
        
    cdef void H3(self, int_num qubit):
        self._c_state.H3(qubit)
        
    cdef void H4(self, int_num qubit):
        self._c_state.H4(qubit)
        
    cdef void H5(self, int_num qubit):
        self._c_state.H5(qubit)
        
    cdef void H6(self, int_num qubit):
        self._c_state.H6(qubit)
        
    cdef void F1(self, int_num qubit):
        self._c_state.F1(qubit)
        
    cdef void F2(self, int_num qubit):
        self._c_state.F2(qubit)
        
    cdef void F3(self, int_num qubit):
        self._c_state.F3(qubit)
        
    cdef void F4(self, int_num qubit):
        self._c_state.F4(qubit)
        
    cdef void F1d(self, int_num qubit):
        self._c_state.F1d(qubit)

    cdef void F2d(self, int_num qubit):
        self._c_state.F2d(qubit)
        
    cdef void F3d(self, int_num qubit):
        self._c_state.F3d(qubit)
        
    cdef void F4d(self, int_num qubit):
        self._c_state.F4d(qubit)
        
    cdef void I(self, int_num qubit):
        pass
        
    cdef void X(self, int_num qubit):
        self._c_state.bitflip(qubit)
        
    cdef void Y(self, int_num qubit):
        self._c_state.Y(qubit)
        
    cdef void Z(self, int_num qubit):
        self._c_state.phaseflip(qubit)
        
    cdef void S(self, int_num qubit):
        self._c_state.phaserot(qubit)
        
    cdef void Sd(self, int_num qubit):
        self._c_state.Sd(qubit)
        
    cdef void R(self, int_num qubit):
        self._c_state.R(qubit)

    cdef void Rd(self, int_num qubit):
        self._c_state.Rd(qubit)
        
    cdef void Q(self, int_num qubit):
        self._c_state.Q(qubit)
        
    cdef void Qd(self, int_num qubit):
        self._c_state.Qd(qubit)
        
    cdef void cnot(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]
        
        self._c_state.cnot(cqubit, tqubit)
        
    cdef void cy(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]
        
        self._c_state.phaserot(tqubit)
        self._c_state.cnot(cqubit, tqubit)
        self._c_state.Sd(tqubit)
        
    cdef void cz(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]
        
        self._c_state.hadamard(tqubit)
        self._c_state.cnot(cqubit, tqubit)
        self._c_state.hadamard(tqubit)
        
    cdef void g2(self, tuple qubits):
        cdef int_num cqubit = qubits[0]
        cdef int_num tqubit = qubits[1]
        
        self._c_state.hadamard(cqubit)
        self._c_state.cnot(tqubit, cqubit)
        self._c_state.cnot(cqubit, tqubit)
        self._c_state.hadamard(tqubit)
        
    cdef void swap(self, tuple qubits):
        cdef int_num qubit1 = qubits[0]
        cdef int_num qubit2 = qubits[1]
        
        self._c_state.swap(qubit1, qubit2)
        
    cdef void II(self, tuple qubits):
        pass
        
    # cpdef unsigned int measure(self, const s.int_num qubit, 
    def measure(self, const int_num qubit, int random_outcome=-1):
        return self._c_state.measure(qubit, random_outcome)
    
    def measureX(self, const int_num qubit, int random_outcome=-1):
        
        cdef unsigned int result
        
        self._c_state.hadamard(qubit)
        result = self._c_state.measure(qubit, random_outcome)
        self._c_state.hadamard(qubit)
        return result
    
    def measureY(self, const int_num qubit, int random_outcome=-1):
        
        cdef unsigned int result
        
        self._c_state.H5(qubit)
        result = self._c_state.measure(qubit, random_outcome)
        self._c_state.H5(qubit)
        return result
    
    cdef void initzero(self, const int_num qubit):
        cdef unsigned int result
        result = self._c_state.measure(qubit, force=0)
        
        if result:
            self._c_state.bitflip(qubit)
            
    cdef void initone(self, const int_num qubit):
        cdef unsigned int result
        result = self._c_state.measure(qubit, force=0)
        
        if not result:
            self._c_state.bitflip(qubit)
            
    cdef void initplus(self, const int_num qubit):
        self.initzero(qubit)
        self._c_state.hadamard(qubit)
        
    cdef void initminus(self, const int_num qubit):
        self.initone(qubit)
        self._c_state.hadamard(qubit)
        
    cdef void initplusi(self, const int_num qubit):
        self.initzero(qubit)
        self._c_state.H5(qubit)
        
    cdef void initminusi(self, const int_num qubit):
        self.initone(qubit)
        self._c_state.H5(qubit)
        
    def logical_sign(self, logical_op, delogical_op=None):
        """

        Args:
            logical_op:
            delogical_op:

        Returns:

        """
        return find_logical_signs(self, logical_op, delogical_op)
            
    def run_gate(self, symbol, locations, **gate_kwargs):
        """

        Args:
            symbol:
            locations:
            **gate_kwargs:

        Returns:

        """

        output = {}
        for location in locations:
            results = self.gate_dict[symbol](self, location, **gate_kwargs)
            if results:
                output[location] = results

        return output

    def run_circuit(self, circuit):
        """

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """

        results = []

        for symbol, locations, gate_kwargs in circuit.items(params=True):
            gate_output = self.run_gate(symbol, locations, **gate_kwargs)
            results.append(gate_output)
                
        return results
    
    @property
    def signs_minus(self):
        return self._c_state.signs_minus
    
    @property
    def signs_i(self):
        return self._c_state.signs_i
    
    @property
    def stabs(self):
        return self._c_state.stabs
    
    @property
    def destabs(self):
        return self._c_state.destabs
        
    def _pauli_sign(self, gen, i_gen):

        if i_gen in self.signs_minus:
            if i_gen in self.signs_i:
                sign = '-i'
            else:
                sign = ' -'
        else:

            if i_gen in self.signs_i:
                sign = ' i'
            else:
                sign = '  '

        return sign
    
    def force_output(self, int_num qubit, forced_output=-1):
        """
        Outputs value.
    
        Used for error generators to generate outputs when replacing measurements.
    
        Args:
            state:
            qubit:
            output:
    
        Returns:
    
        """
        return forced_output

    def col_string(self, gen, print_signs=True):
        """
        Prints out the stabilizers for the column-wise sparse representation.

        :param num_qubits:
        :return:
        """

        col_x = gen['col_x']
        col_z = gen['col_z']

        result = []

        num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            if print_signs:
                sign = self._pauli_sign(gen, i_gen)
            else:
                sign = '  '

            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if i_gen in col_x[qubit] and i_gen not in col_z[qubit]:
                    letter = 'X'

                elif i_gen not in col_x[qubit] and i_gen in col_z[qubit]:
                    letter = 'Z'

                elif i_gen in col_x[qubit] and i_gen in col_z[qubit]:
                    letter = 'W'

                elif i_gen not in col_x[qubit] and i_gen not in col_z[qubit]:
                    letter = 'I'

                stab_letters.append(letter)

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result
    
    def get_largest_degree(self):
        
        cdef:
            list size_stabs_col_x = []
            list size_stabs_col_z = []
            list size_destabs_col_x = []
            list size_destabs_col_z = []
            list size_stabs_row_x = []
            list size_stabs_row_z = []
            list size_destabs_row_x = []
            list size_destabs_row_z = []
            dict output_dict = {}
        
        for col in self._c_state.stabs.col_x:
            size_stabs_col_x.append(len(col))
            
        for col in self._c_state.stabs.col_z:
            size_stabs_col_z.append(len(col))
            
        for col in self._c_state.destabs.col_x:
            size_destabs_col_x.append(len(col))
            
        for col in self._c_state.destabs.col_z:
            size_destabs_col_z.append(len(col))
            
        for row in self._c_state.stabs.row_x:
            size_stabs_row_x.append(len(row))
            
        for row in self._c_state.stabs.row_z:
            size_stabs_row_z.append(len(row))
            
        for row in self._c_state.destabs.row_x:
            size_destabs_row_x.append(len(row))
            
        for row in self._c_state.destabs.row_z:
            size_destabs_row_z.append(len(row))
            
        output_dict = {
            'size_stabs_col_x': size_stabs_col_x,
            'size_stabs_col_z': size_stabs_col_z,
            'size_destabs_col_x': size_destabs_col_x,
            'size_destabs_col_z': size_destabs_col_z,
            'size_stabs_row_x': size_stabs_row_x,
            'size_stabs_row_z': size_stabs_row_z,
            'size_destabs_row_x': size_destabs_row_x,
            'size_destabs_row_z': size_destabs_row_z,
            'size_signs_minus': len(self._c_state.signs_minus),
            'size_signs_i': len(self._c_state.signs_i),
                }
        
        return output_dict
    
    def set_stabs_destabs(self, stabs_row_x, stabs_row_z, destabs_row_x, destabs_row_z):
        
        if len(stabs_row_x) != self.num_qubits:
            raise Exception('Size of `stabs_row_x` must equal `num_qubits`.')
            
        if len(stabs_row_z) != self.num_qubits:
            raise Exception('Size of `stabs_row_z` must equal `num_qubits`.')
        
        if len(destabs_row_x) != self.num_qubits:
            raise Exception('Size of `destabs_row_x` must equal `num_qubits`.')
            
        if len(destabs_row_z) != self.num_qubits:
            raise Exception('Size of `destabs_row_z` must equal `num_qubits`.')
        
        self._c_state.stabs.row_x = stabs_row_x
        self._c_state.stabs.row_z = stabs_row_z
        self._c_state.destabs.row_x = destabs_row_x
        self._c_state.destabs.row_z = destabs_row_z
        
        stabs_col_x = [set() for i in range(self.num_qubits)]
        stabs_col_z = [set() for i in range(self.num_qubits)]
        destabs_col_x = [set() for i in range(self.num_qubits)]
        destabs_col_z = [set() for i in range(self.num_qubits)]
        
        for s, q_set in enumerate(stabs_row_x):
            for q in q_set:
                # stabs_row_x[q].add(s)
                stabs_col_x[q].add(s)
                
        for s, q_set in enumerate(stabs_row_z):
            for q in q_set:
                # stabs_row_z[q].add(s)
                stabs_col_z[q].add(s)

        for s, q_set in enumerate(destabs_row_x):
            for q in q_set:
                # destabs_row_x[q].add(s)
                destabs_col_x[q].add(s)
                
        for s, q_set in enumerate(destabs_row_z):
            for q in q_set:
                # destabs_row_z[q].add(s)
                destabs_col_z[q].add(s)
                
        self._c_state.stabs.col_x = stabs_col_x
        self._c_state.stabs.col_z = stabs_col_z
        self._c_state.destabs.col_x = destabs_col_x
        self._c_state.destabs.col_z = destabs_col_z

    def print_tableau(self, gen, verbose=True, print_signs=True):
        """
        Prints out the stabilizers.
        :return:
        """

        col_str = self.col_string(gen, print_signs=print_signs)
        row_str = self.row_string(gen, print_signs=print_signs)

        if col_str != row_str:
            print('col')
            for line in col_str:
                print(line)

            print('\nrow')
            for line in row_str:
                print(line)

            raise Exception('Something bad happened! String representation of the row-wise vs column-wile '
                            'spare stabilizers do not match!')

        if verbose:
            for line in col_str:
                print(line)

        return col_str
    
    def print_stabs(self):
        str_s = self.print_tableau(self.stabs, verbose=True)
        print('-------------------------------')
        str_d = self.print_tableau(self.destabs, verbose=True, 
                                   print_signs=False)
        
        return str_s, str_d

    def row_string(self, gen, print_signs=True):
        """
        Prints out the stabilizers for the row-wise sparse representation.

        :param num_qubits:
        :return:
        """

        row_x = gen['row_x']
        row_z = gen['row_z']

        result = []

        num_qubits = self.num_qubits

        for i_gen in range(num_qubits):

            stab_letters = []

            # ---- Signs ---- #
            if print_signs:
                sign = self._pauli_sign(gen, i_gen)
            else:
                sign = '  '
            stab_letters.append(sign)

            # ---- Paulis ---- #
            for qubit in range(num_qubits):

                letter = 'U'

                if qubit in row_x[i_gen] and qubit not in row_z[i_gen]:
                    letter = 'X'

                elif qubit not in row_x[i_gen] and qubit in row_z[i_gen]:
                    letter = 'Z'

                elif qubit in row_x[i_gen] and qubit in row_z[i_gen]:
                    letter = 'W'

                elif qubit not in row_x[i_gen] and qubit not in row_z[i_gen]:
                    letter = 'I'

                stab_letters.append(letter)

            # print(''.join(stab_letters))
            result.append(''.join(stab_letters))

        return result
    
    def __dealloc__(self):
        del self._c_state
    