from . import boilerplate
from . import differential_equations
from . import pauli_basis
import numpy as np
import scipy
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from functools import partial
from tqdm import tqdm
from qiskit.circuit.library import standard_gates
from scipy.linalg import null_space
import sys

complex_unitary_log=boilerplate.complex_unitary_log
QFT=boilerplate.qft





#################################################################################
#Create setup for solving geodesic problem in case where q!=1. 

#RHS of evolution equation for initial tangent vectors.
def rhs(solver_params,q,Ham_vec):
    global last_q
    global pbar
    if q > last_q:                # avoid backward / duplicate steps
        pbar.update(q - last_q)
        last_q = q

    N_qubits=solver_params[0]
    N_t=solver_params[1]

    n=int(np.sqrt(len(Ham_vec)))
    
    f,g=boilerplate.get_F_G_matrices(N_qubits,q)
    
    Ham=np.reshape(Ham_vec,(n,n))
    

    #Given the initial hamiltonian for this q, solve the geodesic-schrodinger equation
    sol=differential_equations.solve_matrix_ivp(Ham,N_qubits,q,0,1,N_t)
    time=sol[0]
    H_o_t=sol[1]
    U_o_t=sol[2]
    
    dt=time[1]
    
    #Get Udagger
    U_dagger_o_t=np.transpose(U_o_t,(0,2,1)).conj()

    #Given solution to geodesic-schrodinger equation, computer the jacobi propagator
    A=boilerplate.cal_A(f,g,H_o_t,q) 
        
    K_prop=boilerplate.compute_propagator(A,dt)
    kshape=int(np.sqrt(np.shape(K_prop)[1]))
    k_tensor=np.reshape(K_prop,(K_prop.shape[0],kshape,kshape,kshape,kshape))
    
    J_prop=np.trapezoid(np.einsum('iad,idekl,ieb->iabkl',U_dagger_o_t,k_tensor,U_o_t),time,axis=0)

    
    J_mat=np.reshape(J_prop,(kshape**2,kshape**2))

    J_mat_cond=np.linalg.cond(J_mat)

    tol=1e12

    if J_mat_cond>=tol:
        print(f'Warning, poorly conditioned Jacobi propapgator. Condition number larger than {tol}')
        sys.exit()
        



    Jinv=np.linalg.inv(J_mat)


    tensor_dim=int(np.sqrt(Jinv.shape[0]))
    
    Jinv_tensor=np.reshape(Jinv,(tensor_dim,tensor_dim,tensor_dim,tensor_dim))
    
    if q==1:
        jacobi_arg=differential_equations.approximate_matrix_integral(time,H_o_t,U_o_t)

        dhdq=np.einsum('ijkl,kl',Jinv_tensor,jacobi_arg)
    else:
        jacobi_arg=boilerplate.cal_G(Ham,q)

        dhdq=(np.einsum('ijkl,kl',Jinv_tensor,jacobi_arg)-jacobi_arg)/(q*(q-1))
        
    rhs_value=dhdq.flatten()
            

    return rhs_value





#Define function which solves the initial value problem specified by the rhs function and initial conditions
def solve_ivp(
    h_nought,
    n_qubits,
    n_points,
    rhs_partial,
    rtol,
    atol
):


    q0=1.00
    qf=4**(n_qubits)
    
    global last_q
    global pbar
    try:
        pbar.close
    except:
        pass
    pbar = tqdm(total=qf - q0)
    last_q = q0

    if h_nought.shape[0] != (1 << n_qubits):

        raise ValueError
    
    y0 =h_nought.flatten()
    
    result = scipy.integrate.solve_ivp(
        rhs_partial,
        t_span=(q0,float(qf)),
        y0=y0,
        t_eval=np.logspace(np.log10(q0), np.log10(qf-1e-6), n_points),
        method='RK45',
        rtol=rtol,
        atol=atol
    )

    h_values = result.y
    
    h_mat=np.reshape(h_values,(n_points,2**n_qubits,2**n_qubits))

    pbar.close()
    
    
    

    return result.t, h_values


def ComplexityVQ(sol,solver_params):
    #Calculate complexity vs. the penalty factor based on solution

    N_qubits=solver_params[0]
    N_t=solver_params[1]

    cn3_data=[]

    coeff_data_P=[]
    coeff_data_Q=[]

    ham_hist=sol[1]
    q_vals=sol[0]

    #Count the number of two body terms
    NP=0
    for signature in boilerplate.generate_pauli_signatures(N_qubits):
        if signature[0]<=2:
            NP+=1


    for i,h in enumerate(ham_hist.T):
        ham_final=np.reshape(h,(2**N_qubits,2**N_qubits))
        
        #print(ham_final)
        
        sold=differential_equations.solve_matrix_ivp(ham_final,N_qubits,q_vals[i],0,1,N_t)

        time=sold[0]
        H_o_t=sold[1]
        
        
        coeff_datum=[]
        for signature in boilerplate.generate_P_signatures(n_qubits=N_qubits):
            coeff, basis_matrix = boilerplate.get_coeff_and_basis(H_o_t[-1], pauli_signature=signature)
            coeff_datum.append(coeff)
        coeff_data_P.append(coeff_datum)
        
        coeff_datum=[]
        for signature in boilerplate.generate_pauli_signatures(N_qubits):
            coeff, basis_matrix = boilerplate.get_coeff_and_basis(H_o_t[-1], signature[1])
            weight=N_qubits-signature.count('I')
            if signature[0]>2:
                coeff_datum.append(coeff)
            
        coeff_data_Q.append(coeff_datum)
        

        integrand=[]
        for ht in H_o_t:
            integrand.append(np.sqrt(np.trace(ht@boilerplate.cal_G(ht,q_vals[i]))/2**N_qubits))
        c3=np.trapezoid(integrand,time)
        #c3=np.real(np.sqrt(np.trace(ham_final@boilerplate.cal_G(ham_final,q_vals[i]))))/2**N_qubits
        cn3_data.append(c3)

    coeff_data_P_bottom=coeff_data_P
    return cn3_data, coeff_data_P, coeff_data_Q

def TimeEvolution(solq,Hp,solver_params):
    #Calculate the unitary evolution along the final geodesic, and calculate residual vs time. 

    Ut_new=scipy.linalg.expm(-1j*Hp)

    H_coeff_data_P=[]
    H_coeff_data_Q=[]
    
    ham_hist=solq[1]
    N_qubits=solver_params[0]
    N_t=solver_params[1]



    ham_final=np.reshape(ham_hist.T[-1],(2**N_qubits,2**N_qubits))

    sol=differential_equations.solve_matrix_ivp(ham_final,N_qubits,4**N_qubits-1,0,1,N_t)

    time=sol[0]
    H_o_t=sol[1]
    U_o_t=sol[2]

    U_norm_data=[]


    p_sigs=[]
    for sig in boilerplate.generate_P_signatures(n_qubits=N_qubits):
        p_sigs.append(sig)


    for i,ht in enumerate(H_o_t):
        
        
        coeff_datum=[]
        for signature in boilerplate.generate_P_signatures(n_qubits=N_qubits):
            coeff, basis_matrix = boilerplate.get_coeff_and_basis(ht, pauli_signature=signature)
            coeff_datum.append(coeff)
        H_coeff_data_P.append(coeff_datum)
        
        coeff_datum=[]
        for signature in boilerplate.generate_pauli_signatures(N_qubits):
            coeff, basis_matrix = boilerplate.get_coeff_and_basis(ht, signature[1])
            weight=N_qubits-signature[1].count('i')

            if signature[1] not in p_sigs:
            
                coeff_datum.append(coeff)
        H_coeff_data_Q.append(coeff_datum)
        
        delta_U=Ut_new-U_o_t[i]

        
        
        U_norm_data.append(np.sqrt(np.trace(delta_U@delta_U.T.conj())/2**N_qubits))

    return H_coeff_data_P, H_coeff_data_Q, U_norm_data,time


class solution_object:
        pass


def GeoComplexity(qc,Nt=101,Nq=100,rtol=1e-3,atol=1e-6):
    
    #Get the operator form of the quantum circuit
    operator = Operator(qc)
    # Get the matrix representation
    U_target = operator.data
    #Get the Hamiltonian for the q=1 t=0 case.
    Ham=complex_unitary_log(U_target)

    #Find the number of qubits
    N_qubits=qc.num_qubits

    #Subtract constant term which contributes a global phase. This is our initial condition
    Hp=Ham-np.trace(Ham)*np.identity(2**N_qubits)/2**N_qubits 


    solver_params=[N_qubits,Nt]
    rhs_partial=partial(rhs,solver_params)
    #Calculate the solution
    sol=solve_ivp(Hp,N_qubits,Nq,rhs_partial,rtol,atol)

    #Evolve complexity factor forward to final value
    ###############################################################################
    q_vals=sol[0]
    ComplexityHist,coeff_hist_p,coeff_data_Q=ComplexityVQ(sol,solver_params)

    #Baseline complexity
    final_complexity=ComplexityHist[-1]

    #For highest penalty factor, do the time evolution of the hamiltonian.
    ###############################################################################

    H_coeff_data_P, H_coeff_data_Q, U_norm_data,time=TimeEvolution(sol,Hp,solver_params)

    

    solution=solution_object()

    solution.geocomplex=np.real(final_complexity)
    solution.complexHist=ComplexityHist
    solution.coeffHistQ=coeff_data_Q
    solution.coeffHistP=coeff_hist_p
    solution.q_vals=q_vals
    solution.H_coeff_data_p=H_coeff_data_P
    solution.H_coeff_data_Q=H_coeff_data_Q
    solution.U_norm_data=U_norm_data
    solution.time=time

    return solution


def GateSumComplexity(qc,Nt=101,Nq=100):
    complexitySum=0
    for instr, qargs, cargs in qc.data:
        #print(instr.name)
        # Create a new circuit with only the qubits needed for this gate
        num_qubits = len(qargs)

        if num_qubits==1:
            sub_circ = QuantumCircuit(num_qubits)

            # Map the original qubits to the new sub-circuit qubits
            qubit_map = {qargs[i]: sub_circ.qubits[i] for i in range(num_qubits)}

            # Apply the gate to the mapped qubits
            sub_circ.append(instr, [qubit_map[q] for q in qargs], [])
            complexitySum+=GeoComplexity(sub_circ,Nt,Nq).geocomplex
        elif num_qubits==2:

            if instr.name=='cz':
                complexitySum+=np.sqrt(3)*np.pi/4
            else:
                sub_circ = QuantumCircuit(num_qubits+1)

                # Map the original qubits to the new sub-circuit qubits
                qubit_map = {qargs[i]: sub_circ.qubits[i] for i in range(num_qubits)}

                # Apply the gate to the mapped qubits
                sub_circ.append(instr, [qubit_map[q] for q in qargs], [])
                complexitySum+=GeoComplexity(sub_circ,Nt,Nq).geocomplex

        

        

    return complexitySum

def random_IBM_circuit(num_qubits,num_gates,seed):
    
    gates=['sx','cz','x','rz']

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)
    circ = QuantumCircuit(num_qubits)

    instructions = {
    "i": (standard_gates.IGate(), 1),
    "x": (standard_gates.XGate(), 1),
    "y": (standard_gates.YGate(), 1),
    "z": (standard_gates.ZGate(), 1),
    "h": (standard_gates.HGate(), 1),
    "s": (standard_gates.SGate(), 1),
    "sdg": (standard_gates.SdgGate(), 1),
    "sx": (standard_gates.SXGate(), 1),
    "sxdg": (standard_gates.SXdgGate(), 1),
    "cx": (standard_gates.CXGate(), 2),
    "cy": (standard_gates.CYGate(), 2),
    "cz": (standard_gates.CZGate(), 2),
    "swap": (standard_gates.SwapGate(), 2),
    "iswap": (standard_gates.iSwapGate(), 2),
    "ecr": (standard_gates.ECRGate(), 2),
    "dcx": (standard_gates.DCXGate(), 2),
    }

    np.random.seed(seed)
    for name in samples:
        if name=='rz':
            gate=standard_gates.RZGate(phi=2*np.pi*np.random.rand())
            nqargs=1
            
        else:
            gate, nqargs = instructions[name]

        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs, copy=False)

    return circ


def convergenceTester(qc):

    # 4 values for Nt (must be odd between 101 and 201)
    Nt_values = np.linspace(101, 201, 4, dtype=int)
    Nt_values = Nt_values + (Nt_values % 2 == 0)  # ensure odd

    # 4 values for Nq between 100 and 200
    Nq_values = np.linspace(100, 200, 4, dtype=int)

    # Create grid
    Nt_grid, Nq_grid = np.meshgrid(Nt_values, Nq_values)
    complexity_values = np.zeros_like(Nt_grid, dtype=float)

    for i in range(len(Nq_values)):
        for j in range(len(Nt_values)):
            complexity_values[i, j] = GeoComplexity(qc,Nt_grid[i, j], Nq_grid[i, j]).geocomplex

    return complexity_values, Nt_grid, Nq_grid


def NaiveComplexity(qc):

    #Get the operator form of the quantum circuit
    operator = Operator(qc)
    # Get the matrix representation
    U_target = operator.data
    #Get the Hamiltonian for the q=1 t=0 case.
    Ham=complex_unitary_log(U_target)

    #Find the number of qubits
    N_qubits=qc.num_qubits

    #Subtract constant term which contributes a global phase. This is our initial condition
    Hp=Ham-np.trace(Ham)*np.identity(2**N_qubits)/2**N_qubits 

    return np.sqrt(np.trace(Hp@boilerplate.cal_G(Hp,4**N_qubits))/2**N_qubits)





