#%% Imports
import numpy as np
import jax.numpy as jnp
import jax

#%%
def elementExtraction(matrice, k):
    return jax.vmap(lambda ligne: ligne[k])(matrice)

def listElementExtraction(matrice, indices):
    return jax.vmap(lambda k: jax.vmap(lambda ligne: ligne[k])(matrice))(indices)

# H = 1
# Sigma_s_0 = 0.05


# def yieldFunction(sigma_elem, p_xi):
#     eqVM = equivalentVM(sigma_elem)
#     return eqVM - Sigma_s_0*(1 + H * p_xi)

# def testYieldFunction(sigma_elem, p_xi):
#     return yieldFunction(sigma_elem, p_xi) > 0


def deviatoric(array):
    trace = sum(array[:3])
    return array - (1/3*trace*np.array([1, 1, 1, 0, 0, 0]))[:,np.newaxis]

# def equivalentVM(dev_sigma_elem):
#     # dev_s = deviatoric(sigma_elem[:,np.newaxis])
#     # print(dev_s)
#     # print(dev_s.shape)
#     vm = jnp.sum(dev_sigma_elem ** 2)
#     # vm = (sigma_elem.T @ sigma_elem)[0,0]
#     return jnp.sqrt(3/2 * vm)

# def yieldFunction(vmstress, p_xi, H, S0):
#     nQP = vmstress.shape[0]
#     return vmstress - S0*(np.ones(nQP) + H*p_xi) 

# def testPlasticity(Sigma, lpxi, nQP, H, S0):
#     s_matrix = Sigma.reshape((6, nQP))
#     lnQP = jnp.arange(0, nQP, 1)
#     s_elem = listElementExtraction(s_matrix, lnQP)
#     s_dev_elem = jax.vmap(deviatoric)(s_elem)
#     vmstress = jax.vmap(equivalentVM)(s_dev_elem)
#     f_sigma = yieldFunction(vmstress, lpxi, H, S0) 
#     return s_elem, vmstress, f_sigma


def equivalentVM(sigma_elem):
    dev_s = deviatoric(sigma_elem[:,np.newaxis])
    # print(dev_s.shape)
    vm = jnp.sum(dev_s ** 2) + jnp.sum(dev_s[3:] ** 2)
    # vm = (sigma_elem.T @ sigma_elem)[0,0]
    return jnp.sqrt(3/2 * vm)

def yieldFunction(vmstress, p_xi, H, S0):
    nQP = vmstress.shape[0]
    return vmstress - S0*(np.ones(nQP) + H*p_xi) 

# def testPlasticity(Sigma, lpxi):
#     s_matrix = Sigma.reshape((6, nQP))
#     lnQP = jnp.arange(0, nQP, 1)
#     s_elem = listElementExtraction(s_matrix, lnQP)
#     devs_elem = deviatoric(s_elem)
#     vmstress = jax.vmap(equivalentVM)(devs_elem)
#     lpxi = jnp.zeros(nQP)
#     f_sigma = yieldFunction(vmstress, lpxi) 
#     return f_sigma


def testPlasticity(Sigma, lpxi, nQP, H, S0):
    s_matrix = Sigma.reshape((6, nQP))
    lnQP = jnp.arange(0, nQP, 1)
    s_elem = listElementExtraction(s_matrix, lnQP)
    vmstress = jax.vmap(equivalentVM)(s_elem)
    f_sigma = yieldFunction(vmstress, lpxi, H, S0) 
    return s_elem, vmstress, f_sigma


# def fp(XiEl, pxi, H, S0):
#     S_el = Rxi @ XiEl  # - Xi_th_loc
#     trace = sum(S_el[:3])
#     devS_el = S_el - 1/3*trace*jnp.array([1, 1, 1, 0, 0, 0])[:, np.newaxis]
#     vm = jnp.sum(devS_el ** 2)
#     eqVM = jnp.sqrt(3/2 * vm)
#     return eqVM - Sigma_s_0*(1 + H*pxi) 