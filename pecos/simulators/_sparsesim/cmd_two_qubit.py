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

from typing import Tuple
from .state import SparseSim
from .cmd_one_qubit import Q, S, Sd, R, Rd


def CNOT(state: SparseSim,
         qubits: Tuple[int, int]) -> None:
    """

    II -> II
    XI -> XX
    ZI -> ZI
    WI -> WX
    IX -> IX
    IZ -> ZZ
    IW -> ZW
    XX -> XI
    XZ -> WW
    XW -> WZ
    ZX -> ZX
    ZZ -> IZ
    ZW -> IW
    WX -> WI
    WZ -> XW
    WW -> XZ

    II -> II
    XI -> XX
    ZI -> ZI
    YI -> YX
    IX -> IX
    IZ -> ZZ
    IY -> ZY
    XX -> XI
    XZ -> -YY
    XY -> YZ
    ZX -> ZX
    ZZ -> IZ
    ZY -> IY
    YX -> YI
    YZ -> XY
    YY -> -XZ

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    set_q1 = {qubit1}
    set_q2 = {qubit2}

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # X2 += X1 for rows
        for i in g.col_x[qubit1]:
            g.row_x[i].symmetric_difference_update(set_q2)

        # Z1 += Z2 for rows
        for i in g.col_z[qubit2]:
            g.row_z[i].symmetric_difference_update(set_q1)

        # Update the X column
        # X2 += X1
        g.col_x[qubit2] ^= g.col_x[qubit1]

        # Update the Z column
        # Z1 += Z2
        g.col_z[qubit1] ^= g.col_z[qubit2]


def CZ(state: SparseSim,
       qubits: Tuple[int, int]) -> None:
    """
    Applies a Controlled-Z gate (CZ) rotation.

    This version is best for a large number of qubits (aboa ut >= 150)

    II -> II
    XI -> XZ
    ZI -> ZI
    WI -> WZ
    IX -> ZX
    IZ -> IZ
    IW -> ZW
    XX -> -WW
    XZ -> XI
    XW -> -WX
    ZX -> IX
    ZZ -> ZZ
    ZW -> IW
    WX -> -XW
    WZ -> WI
    WW -> -XX

    II -> II
    XI -> XZ
    ZI -> ZI
    YI -> YZ
    IX -> ZX
    IZ -> IZ
    IY -> ZY
    XX -> YY
    XZ -> XI
    XY -> -YX
    ZX -> IX
    ZZ -> ZZ
    ZY -> IY
    YX -> -XY
    YZ -> YI
    YY -> XX

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    stabs = state.stabs

    # Change the sign appropriately
    stabs.signs_minus ^= stabs.col_x[qubit1] & stabs.col_x[qubit2]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:

        old_z1_col = set(g.col_z[qubit1])
        old_z2_col = set(g.col_z[qubit2])

        # Update columns

        # Z1 += X2
        g.col_z[qubit1] ^= g.col_x[qubit2]

        # Z2 += X1
        g.col_z[qubit2] ^= g.col_x[qubit1]

        # Update rows

        # Z gens added
        z1_added = g.col_z[qubit1] - old_z1_col
        z2_added = g.col_z[qubit2] - old_z2_col

        for i in z1_added:
            g.row_z[i].add(qubit1)

        for i in z2_added:
            g.row_z[i].add(qubit2)

        del z1_added
        del z2_added

        # Z gens removed
        z1_removed = old_z1_col - g.col_z[qubit1]
        z2_removed = old_z2_col - g.col_z[qubit2]

        for i in z1_removed:
            g.row_z[i].discard(qubit1)

        for i in z2_removed:
            g.row_z[i].discard(qubit2)


def CY(state: SparseSim,
       qubits: Tuple[int, int]) -> None:
    """
    Applies a Controlled-Y gate

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    _, qubit2 = qubits

    S(state, qubit2)
    CNOT(state, qubits)
    Sd(state, qubit2)


def SWAP(state: SparseSim,
         qubits: Tuple[int, int]) -> None:
    """
    Applies a SWAP gate to the generators

    II -> II
    XI -> IX
    ZI -> IZ
    WI -> IW
    IX -> XI
    IZ -> ZI
    IW -> WI
    XX -> XX
    XZ -> ZX
    XW -> WX
    ZX -> XZ
    ZZ -> ZZ
    ZW -> WZ
    WX -> WX
    WZ -> ZW
    WW -> WW

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:

        # Update rows

        # Swap the Xs of qubit1 and qubit2 for rows

        # set of gens with Xs on qubit1 but not qubit2
        xin1 = g.col_x[qubit1] - g.col_x[qubit2]

        # set of gens with Xs on qubit2 but not qubit1
        xin2 = g.col_x[qubit2] - g.col_x[qubit1]

        for i in xin1:
            g.row_x[i].discard(qubit1)
            g.row_x[i].add(qubit2)

        for i in xin2:
            g.row_x[i].discard(qubit2)
            g.row_x[i].add(qubit1)

        del xin1
        del xin2

        # Swap the Zs of qubit1 and qubit2 for cols

        # set of gens with Zs on qubit1 but not qubit2
        zin1 = g.col_z[qubit1] - g.col_z[qubit2]

        # set of gens with Zs on qubit2 but not qubit1
        zin2 = g.col_z[qubit2] - g.col_z[qubit1]

        for i in zin1:
            g.row_z[i].discard(qubit1)
            g.row_z[i].add(qubit2)

        for i in zin2:
            g.row_z[i].discard(qubit2)
            g.row_z[i].add(qubit1)

        del zin1
        del zin2

        # Update columns

        # Swap the Xs of qubit1 and qubit2 for cols
        g.col_x[qubit1], g.col_x[qubit2] = g.col_x[qubit2], g.col_x[qubit1]

        # Swap the Zs of qubit1 and qubit2 for cols
        g.col_z[qubit1], g.col_z[qubit2] = g.col_z[qubit2], g.col_z[qubit1]


def G2(state: SparseSim,
       qubits: Tuple[int, int]) -> None:
    """
    Applies a CZ.H(1).H(2).CZ to the generators

    This gate has a Schmidt rank of 4

    II -> II
    XI -> IX
    ZI -> XZ
    WI -> XW
    IX -> XI
    IZ -> ZX
    IW -> WX
    XX -> XX
    XZ -> ZI
    XW -> WI
    ZX -> IZ
    ZZ -> -WW
    ZW -> -ZW
    WX -> IW
    WZ -> -WZ
    WW -> ZZ

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    stabs = state.stabs

    # Change the sign appropriately
    # sign += Z1*Z2
    stabs.signs_minus ^= stabs.col_z[qubit1] & stabs.col_z[qubit2]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:

        # set of gens with Zs on qubit1 but not qubit2
        zin1 = g.col_z[qubit1] - g.col_z[qubit2]

        # set of gens with Zs on qubit2 but not qubit1
        zin2 = g.col_z[qubit2] - g.col_z[qubit1]

        old_x1_col = set(g.col_x[qubit1])
        old_x2_col = set(g.col_x[qubit2])

        # Update columns
        # ---------------------

        # Swap X1 and X2 columns
        # Swap the Xs of qubit1 and qubit2 for cols
        g.col_x[qubit1], g.col_x[qubit2] = g.col_x[qubit2], g.col_x[qubit1]

        # X1 += Z1
        g.col_x[qubit1] ^= g.col_z[qubit1]

        # X2 += Z2
        g.col_x[qubit2] ^= g.col_z[qubit2]

        # Swap Z1 and Z2
        # Swap the Zs of qubit1 and qubit2 for cols
        g.col_z[qubit1], g.col_z[qubit2] = g.col_z[qubit2], g.col_z[qubit1]

        # Update the rows
        # ---------------------

        # Swap the Zs of qubit1 and qubit2 for cols

        for i in zin1:
            g.row_z[i].discard(qubit1)
            g.row_z[i].add(qubit2)

        for i in zin2:
            g.row_z[i].discard(qubit2)
            g.row_z[i].add(qubit1)

        del zin1
        del zin2

        # Z gens added
        x1_added = g.col_x[qubit1] - old_x1_col
        x2_added = g.col_x[qubit2] - old_x2_col

        for i in x1_added:
            g.row_x[i].add(qubit1)

        for i in x2_added:
            g.row_x[i].add(qubit2)

        del x1_added
        del x2_added

        # Z gens removed
        x1_removed = old_x1_col - g.col_x[qubit1]
        x2_removed = old_x2_col - g.col_x[qubit2]

        for i in x1_removed:
            g.row_x[i].discard(qubit1)

        for i in x2_removed:
            g.row_x[i].discard(qubit2)

        del x1_removed
        del x2_removed


def II(state: SparseSim,
       qubits: Tuple[int, int]) -> None:
    """
    Two qubit identity.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    pass


def SqrtXX(state: SparseSim,
           qubits: Tuple[int, int]) -> None:
    """
    Applies a square root of XX rotation to generators

    Sqrt XX: XI->XI, IX->IX, ZI->-iWX (-YX), IZ->-iXW (-XY)
    sign rule: if odd # of Zs -> -i
    Pauli rule: if odd # of Zs -> XX

    II -> II
    XI -> XI
    ZI -> -iWX
    WI -> -iZX
    IX -> IX
    IZ -> -iXW
    IW -> -iXZ
    XX -> XX
    XZ -> -iIW
    XW -> -iIZ
    ZX -> -iWI
    ZZ -> ZZ
    ZW -> ZW
    WX -> -iZI
    WZ -> WZ
    WW -> WW

    II -> II
    XI -> XI
    ZI -> -YX
    YI -> ZX
    IX -> IX
    IZ -> -XY
    IY -> XZ
    XX -> XX
    XZ -> -IY
    XY -> IZ
    ZX -> -YI
    ZZ -> ZZ
    ZY -> ZY
    YX -> ZI
    YZ -> YZ
    YY -> YY

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """

    qubit1, qubit2 = qubits

    stabs = state.stabs

    # Change the sign appropriately
    oddzs = stabs.col_z[qubit1] ^ stabs.col_z[qubit2]
    stabs.signs_minus ^= oddzs

    # imaginary:
    # places that have no is but will need them added
    add_is = oddzs - stabs.signs_i

    # places where is already exist
    gens_common = stabs.signs_i & oddzs
    # i * i = -1
    stabs.signs_minus ^= gens_common
    # Remove them from i's
    stabs.signs_i -= gens_common

    # 1*i = i
    stabs.signs_i |= add_is

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:

        old_x1_col = set(g.col_x[qubit1])
        old_x2_col = set(g.col_x[qubit2])

        # Add XX if odd number of Zs
        oddzs = g.col_z[qubit1] ^ g.col_z[qubit2]

        # Update columns
        g.col_x[qubit1] ^= oddzs
        g.col_x[qubit2] ^= oddzs

        # Update rows

        # Z gens added
        x1_added = oddzs - old_x1_col
        x2_added = oddzs - old_x2_col

        for i in x1_added:
            g.row_x[i].add(qubit1)

        for i in x2_added:
            g.row_x[i].add(qubit2)

        del x1_added
        del x2_added

        # Z gens removed
        x1_removed = old_x1_col - g.col_x[qubit1]
        x2_removed = old_x2_col - g.col_x[qubit2]

        for i in x1_removed:
            g.row_x[i].discard(qubit1)

        for i in x2_removed:
            g.row_x[i].discard(qubit2)


def SqrtXX2(state: SparseSim,
            qubits: Tuple[int, int]) -> None:
    """
    Applies a square root of XX rotation to generators

    Sqrt XX: XI->XI, IX->IX, ZI->-iWX (-YX), IZ->-iXW (-XY)
    sign rule: if odd # of Zs -> -i
    Pauli rule: if odd # of Zs -> XX

    II -> II
    XI -> XI
    ZI -> -iWX
    WI -> -iZX
    IX -> IX
    IZ -> -iXW
    IW -> -iXZ
    XX -> XX
    XZ -> -iIW
    XW -> -iIZ
    ZX -> -iWI
    ZZ -> ZZ
    ZW -> ZW
    WX -> -iZI
    WZ -> WZ
    WW -> WW

    II -> II
    XI -> XI
    ZI -> -YX
    YI -> ZX
    IX -> IX
    IZ -> -XY
    IY -> XZ
    XX -> XX
    XZ -> -IY
    XY -> IZ
    ZX -> -YI
    ZZ -> ZZ
    ZY -> ZY
    YX -> ZI
    YZ -> YZ
    YY -> YY

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    
    qubit1, qubit2 = qubits
    Q(state, qubit1)  # Sqrt X
    Q(state, qubit2)  # Sqrt X
    Rd(state, qubit1)  # (Sqrt Y)^\dagger
    CNOT(state, qubits)  # CNOT (why didn't I capitalized this?)
    R(state, qubit1)  # Sqrt Y
