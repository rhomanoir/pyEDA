from pyscf import gto, scf
import numpy as np
import copy
from .ets_nocv import ETS_NOCV, U_ETS_NOCV
from .utils import index_parse, make_frag_geom, timer


class NOCV_Analysis:
    def __init__(self, eda, tol = 0.01):
        self.eda = eda
        self.frag_idx_list = self.get_frag_idx_list()
        self.atom_aos_list = self.get_atom_aos_list()
        self.nao_mat = self.get_nao_mat()
        self.tol = tol
        self.decompose()

    @staticmethod
    def get_real_atom_indices(mol):
        charges = mol.atom_charges()
        return np.where(charges > 0)[0].tolist()

    def report(self):
        HARTREE_TO_KCAL = 627.509474
        HARTREE_TO_KJ = 2625.49962
    
        name_width = 25
        num_width = 18
    
        separator = '-' * (name_width + num_width * 3 + 10)
    
        print(f"\n EDA report")
        print(separator)
        print(f"{'Component':<{name_width}} {'a.u.':>{num_width}} {'kcal/mol':>{num_width}} {'kJ/mol':>{num_width}}")
        print(separator)
    
        for key, value in self.eda.EDA.items():
            val_ha = float(value)
            val_kcal = val_ha * HARTREE_TO_KCAL
            val_kj = val_ha * HARTREE_TO_KJ
        
            print(f"{key:<{name_width}} {val_ha:>{num_width}.8f} {val_kcal:>{num_width}.4f} {val_kj:>{num_width}.4f}")
    
        print(separator)
      

    def decompose(self):
        self.weights = []
        self.idxs_neg = []
        self.idxs_pos = []
        self.keys = []
        self.eigs = []
        self.energies = []
        for key, value in self.eda.NOCV.items():
            NOCV_idxs_neg = np.sort(np.where(value["eigenvalues"]<-self.tol))[0]
            NOCV_idxs_pos = np.sort(np.where(value["eigenvalues"]>self.tol))[0][::-1]
            C_nocv_nao = np.linalg.inv(self.nao_mat) @ value['coefficients_ao']
            weights = C_nocv_nao**2
            self.keys.append(key)
            self.weights.append(weights)
            self.eigs.append(value["eigenvalues"])
            self.energies.append(value["orbital_energies"])
            self.idxs_neg.append(NOCV_idxs_neg)
            self.idxs_pos.append(NOCV_idxs_pos)


    
    def get_orb_nao_details(self, k, orb_idx, threshold=1e-6):

        populations = self.weights[k][:, orb_idx]
        
        ao_labels = self.get_nao_labels(k)
    
        nonzero_mask = np.abs(populations) > threshold
        nonzero_idx = np.where(nonzero_mask)[0]
    
        return (
            nonzero_idx,
            np.array(ao_labels)[nonzero_mask],
            populations[nonzero_mask],
            np.sum(populations)
        )


    def report_orb_detailed(self, k, orb_idx, threshold=1e-6):

        idxs, labels, pops, total = self.get_orb_nao_details(k, orb_idx, threshold)

        labels = [f"{a} {b} {c}{d}{f}" for (a, b, c, d, f) in labels]
        
        print(f"NOCV #{orb_idx} (key={self.keys[k]}), "
              f"eigenvalue = {self.eigs[k][orb_idx]:.6f}, "
              f"energy = {self.energies[k][orb_idx]:.6f}")
        print(f"  AO Index    NAO Label                      Population")
        print("-" * 60)
        
        for i, label, pop in zip(idxs, labels, pops):
            print(f"{i:^12d}{label:^30s}{pop:^15.6f}")
        
        print("-" * 60)
        print(f"{'Total population:':^42s}{total:^15.6f}")
        
        #return idxs, labels, pops

    def get_orb_contribs(self, k, orb_idx):
        result = np.zeros(self.eda.mf_molecule.mol.nao)
        for i in range(len(self.frag_idx_list)):
            ao_idxs = np.where(self.atom_aos_list == i)[0]
            contrib = np.sum(self.weights[k][:, orb_idx][ao_idxs])
            result[i] = contrib
        return result


    def report_orbital(self, k, orb_idx):

        contributions = self.get_orb_contribs(k, orb_idx)
    
        print(f"NOCV #{orb_idx}, eigenvalue = {self.eigs[k][orb_idx]}, energy = {self.energies[k][orb_idx]}")
        print("  N_At    Symb    N_frag    Contrib(%)   ")
        print("----------------------------------------")
    
        for i in range(len(self.frag_idx_list)):
            print(f"{i:^8d}{self.eda.mf_molecule.mol.atom_symbol(i):^8s}"
                  f"{int(self.frag_idx_list[i]):^8}{contributions[i]:^18.4f}")
    

    def get_nao_labels(self, k):
        mol = self.eda.mf_molecule.mol
        mf = self.eda.mf_molecule
        l_list = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
        start_idxs = [1, 2, 3, 4, 5, 6, 7]
        coord_sets = [{''},
                      {'x', 'y', 'z'},
                      {'xy', 'yz', 'z^2', 'xz', 'x2-y2'},
                      {'-3', '-2', '-1', '+0', '+1', '+2', '+3'},
                      {'-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4'},
                      {'-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5'},
                      {'-6', '-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5', '+6'}]
    
        ao_labels = mol.sph_labels(fmt=False)
        ao_labels = [(a, b, c[:-1], c[-1], d) for (a, b, c, d) in ao_labels]
        nao_labels = copy.copy(ao_labels)

        for i in range(len(nao_labels)):
            contribs = np.abs(self.nao_mat[:, i])
            max_idx = np.argmax(contribs)
            nao_labels[i] = ao_labels[max_idx]

        dm_ao = mf.make_rdm1()
        if isinstance(dm_ao, tuple):
            dm_ao = dm_ao[k]
        S = self.eda.mf_molecule.get_ovlp()
        dm_nao = self.nao_mat.T @ S @ dm_ao @ S @ self.nao_mat
        nao_occs = np.diag(dm_nao)

        for i in range(len(self.frag_idx_list)):
            orb_subset_idxs = [j for j in range(len(ao_labels)) if ao_labels[j][0] == i]
            for il, l in enumerate(l_list):
                orb_subsubset_idxs = [k for k in orb_subset_idxs if ao_labels[k][3] == l]
                if not orb_subsubset_idxs:
                    pass
                else:
                    coord_set = coord_sets[il]
                    for jj in coord_set:
                        coord_orb_subsubset_idxs = [k for k in orb_subsubset_idxs if ao_labels[k][4] == jj]
                        sorted_indices = sorted(coord_orb_subsubset_idxs, key=lambda iii: nao_occs[iii], reverse=True)
                        counter = start_idxs[il]
                        for ii in sorted_indices:
                            nao_labels[ii] = (i, mol.atom_symbol(i), counter, l, jj)
                            counter+=1
        return nao_labels

    
    def get_orb_ao_details(self, k, orb_idx, threshold=1e-6):
        
        c_ao = self.eda.NOCV[self.keys[k]]['coefficients_ao'][:, orb_idx]
        
        populations = c_ao ** 2
        
        ao_labels = self.eda.mf_molecule.mol.ao_labels()
        
        nonzero_mask = np.abs(populations) > threshold
        nonzero_idx = np.where(nonzero_mask)[0]
        
        return (
            nonzero_idx,
            np.array(ao_labels)[nonzero_mask],
            c_ao[nonzero_mask],
            np.sum(populations)
        )
    
    def report_orb_detailed_ao(self, k, orb_idx, threshold=1e-6):

        idxs, labels, pops, total = self.get_orb_ao_details(k, orb_idx, threshold)
        
        print(f"NOCV #{orb_idx} (key={self.keys[k]}), "
              f"eigenvalue = {self.eigs[k][orb_idx]:.6f}, "
              f"energy = {self.energies[k][orb_idx]:.6f}")
        print(f"  AO Index    AO Label                       Coefficient")
        print("-" * 60)
        
        for i, label, pop in zip(idxs, labels, pops):
            print(f"{i:^12d}{label:^30s}{pop:^15.6f}")
        
        print("-" * 60)
        #print(f"{'Total population:':^42s}{total:^15.6f}")
        
        #return idxs, labels, pops

    def get_nocv_transfers(self, wmat):
        n_nocv = wmat.shape[1]
        n_fragments = len(self.eda.mf_fragments)
        result = np.zeros((n_fragments, n_nocv))
        ao_to_fragment = self.frag_idx_list.astype(int)[self.atom_aos_list.astype(int)]
        np.add.at(result, ao_to_fragment, wmat)
        frag_ids = np.unique(ao_to_fragment)
        return result, ao_to_fragment, frag_ids

    def report_charge_transfer(self):
        for i in range(len(self.keys)):
            print("\n Charge transfer for "+self.keys[i]+" orbials\n")
            npos = len(self.idxs_pos[i])
            nneg = len(self.idxs_neg[i])
            weights_scaled = self.weights[i] *  self.eigs[i]
            weights_frag, ao_to_fragment, frag_ids = self.get_nocv_transfers(weights_scaled)
            frag_tot = weights_frag.sum(axis=1)
            if nneg != npos:
                print("Warning! NOCV orbitals are not paired!")    
            else:
                print(f"Charge transfer for each pair")
                col_width = 10  
                frag_width = 12 
                header =  f"{'Fragment':>{frag_width}}"
                p_contribs = np.zeros((len(frag_ids), nneg))
                for j in range(nneg):
                    header += f"{'Pair_' + str(j+1):>{col_width}}"
                    ineg = self.idxs_neg[i][j]
                    ipos = self.idxs_pos[i][j]
                    p_contribs[:, j] = weights_frag[:, ineg]+weights_frag[:, ipos]
                separator = "-" * len(header)
                print(header)
                print(separator)

                for j, fid in enumerate(frag_ids):
                    row =  f"{'Frag ' + str(fid):>{frag_width}}"
                    for k in range(nneg):
                        row += f"{p_contribs[j, k]:>{col_width}.4f}"
                    print(row)

            print("Total electron population change:")
            for j in frag_ids:
                print(f"Frag #{j} : {frag_tot[j]}")


    def find_midpoint(self, mol1, mol2, psum1, psum2):
        real_idx1 = self.get_real_atom_indices(mol1)
        real_idx2 = self.get_real_atom_indices(mol2)

        if len(real_idx1) == 0 or len(real_idx2) == 0:
            raise ValueError("One oof the molecules has no real atoms!")

        coords1 = mol1.atom_coords()[real_idx1]
        coords2 = mol2.atom_coords()[real_idx2]
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        min_flat_index = np.argmin(distances)
        idx1_loc, idx2_loc = np.unravel_index(min_flat_index, distances.shape)

        #print("psums", psum1, psum2)

        #idx1 = real_idx1[idx1_loc]
        #idx2 = real_idx2[idx2_loc]

        return coords1[idx1_loc], coords2[idx2_loc], (psum1*coords1[idx1_loc] + psum2*coords2[idx2_loc])/(psum1+psum2)
    
    def get_frag_idxs(self, ao_to_fragment, fragment_id):
        return np.where(np.array(ao_to_fragment) == fragment_id)[0]

    def get_submat_for_frag(self, Mmat, ao_to_fragment, i):
        idx_i = self.get_frag_idxs(ao_to_fragment, i)
        return Mmat[np.ix_(idx_i, idx_i)]

    def get_submat_for_frag_pair(self, Mmat, ao_to_fragment, i, j):
        idx_i = self.get_frag_idxs(ao_to_fragment, i)
        idx_j = self.get_frag_idxs(ao_to_fragment, j)
        return Mmat[np.ix_(idx_i, idx_j)]
    

    def step_perpendicular(self, vector, point, distance):
        n = vector / np.linalg.norm(vector)                
        arb = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0]) 
        u = np.cross(n, arb)                                
        u = u / np.linalg.norm(u)                           
        return point + distance * u 
    
    def step_grad(self, ao_values, P1, point, dist):
        mol = self.eda.mf_molecule.mol
        ao_grad_raw = mol.eval_gto('GTOval_ip', point.reshape(1, -1))
        ao_grads = ao_grad_raw[:, 0, :].T
        grad_rho = 2 * (ao_values @ P1 @ ao_grads)
        norm_grad = np.linalg.norm(grad_rho)
        if norm_grad < 1e-10:
            norm_grad = 1e-10
        direction = grad_rho / norm_grad
        return point + direction * dist




    def get_Pdif_ratio(self, key, orb_idx, Pmat, ao_to_fragment, i, j):

        idx_i = self.get_frag_idxs(ao_to_fragment, i)
        idx_j = self.get_frag_idxs(ao_to_fragment, j)

        P1 = np.zeros(Pmat.shape) 
        P1[np.ix_(idx_i, idx_i)] = Pmat[np.ix_(idx_i, idx_i)]
        P1[np.ix_(idx_j, idx_j)] = Pmat[np.ix_(idx_j, idx_j)]

        P2 = copy.copy(P1)
        P2[np.ix_(idx_i, idx_j)] = Pmat[np.ix_(idx_i, idx_j)]
        P2[np.ix_(idx_j, idx_i)] = Pmat[np.ix_(idx_j, idx_i)]

        mol1 = self.eda.mf_fragments[i].mol
        mol2 = self.eda.mf_fragments[j].mol
        contribs = self.get_orb_contribs(key, orb_idx)
        psum1 = np.sum(contribs[np.where(self.frag_idx_list == i)])
        psum2 = np.sum(contribs[np.where(self.frag_idx_list == j)])
        r1, r2, midp = self.find_midpoint(mol1, mol2, psum1, psum2)

        mol = self.eda.mf_molecule.mol
        ao_values = mol.eval_gto('GTOval', midp.reshape(1, -1))[0]
        dens1 = ao_values @ P1 @ ao_values
        if dens1<0.000001:
            midp = self.step_grad(ao_values, P1, midp, 0.1) #self.step_perpendicular(r2-r1, midp, 0.1)
            ao_values = mol.eval_gto('GTOval', midp.reshape(1, -1))[0]
            dens1 = ao_values @ P1 @ ao_values

        dens2 = ao_values @ P2 @ ao_values

        print("Density at midpoint, no nondiagonal elements: ", dens1)
        print("Density at midpoint with nondiagonal elements: ", dens2)

        result = (dens2-dens1)/np.max([dens1, dens2])

        return result

    def frag_bonding_matrix(self, orb_idx, tol=0.001, spin='restricted'):
        key = 0
        if spin == 'beta':
            key = 1

        c_mo = self.eda.NOCV[self.keys[key]]['coefficients_ao'][:, orb_idx]
        pop_matrix = np.outer(c_mo, c_mo)
                
        ao_to_fragment = self.frag_idx_list.astype(int)[self.atom_aos_list.astype(int)]

        n_frag = len(self.eda.mf_fragments)

        bond_matrix = np.zeros((n_frag, n_frag))

        for i in range(n_frag):
            for j in range(i + 1, n_frag):  
                print("\nFragments ",i, j)
                
                r = self.get_Pdif_ratio(key, orb_idx, pop_matrix, ao_to_fragment, i, j)
                
                bond_matrix[i][j] = bond_matrix[j][i]  =  r

        bond_matrix = bond_matrix/np.max(np.abs(bond_matrix))

        print("\n Fragment Bonding Matrix:")
        print(bond_matrix)        
        
    

    def report_by_atom(self):
        for i in range(len(self.keys)):
            print("\n Atomic contribtions for "+self.keys[i]+" orbials\n")
            npos = len(self.idxs_pos[i])
            nneg = len(self.idxs_neg[i])
            if nneg != npos:
                print("Warning! NOCV orbitals are not paired!")
                idxlist = np.concatenate(self.idxs_neg[i], self.idxs_pos[i])                
                for orb_idx in idxlist:
                    self.report_orbital(i, orb_idx)
            else:
                print(f"Found {npos:^10d} pairs with eigenvalues > {self.tol:^16.5f}\n")
                for j in range(nneg):
                    ineg = self.idxs_neg[i][j]
                    ipos = self.idxs_pos[i][j]
                    en = self.eigs[i][ipos]*(self.energies[i][ipos]-self.energies[i][ineg])
                    print(f"\nPair {j+1} (energy contribution = {en:.8f}):")
                    _ = self.report_orbital(i, ineg)
                    _ = self.report_orbital(i, ipos)



    def get_frag_idx_list(self):
        result = np.zeros(self.eda.mf_molecule.mol.natm)
        for i, item in enumerate(self.eda.mf_fragments):
            result[self.get_real_atom_indices(item.mol)] = i
        return result
    
    def get_atom_aos_list(self):
        result = np.zeros(self.eda.mf_molecule.mol.nao)
        ao_slices = self.eda.mf_molecule.mol.aoslice_by_atom()
        for i in range(len(ao_slices)):
            ao_indices = list(range(ao_slices[i][2], ao_slices[i][3]))
            result[ao_indices] = i
        return result
    
    def get_nao_mat(self):
        from pyscf import lo
        return lo.orth_ao(self.eda.mf_molecule, 'nao')
    
    

