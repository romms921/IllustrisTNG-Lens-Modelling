import glafic
import numpy as np

m = [round(x, 5) for x in np.linspace(0.05, 0.1, 2)]
n = [round(x, 5) for x in np.linspace(0, 360, 100)]
o = [round(x, 5) for x in np.linspace(0, 1, 10)]

constraint_file = '/Volumes/T7 Shield/Simulations/Input/pos_point.dat'
prior_file = None

for i in range(len(m)):
    for j in range(len(n)):
            model_name = f'SIE_SHEAR_{m[i]}_{n[j]}'
            path = f'/Volumes/T7 Shield/Simulations/Output/{model_name}'
            glafic.init(0.3, 0.7, -1.0, 0.7, path, 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb=0)
            glafic.set_secondary('chi2_splane 1', verb=0)
            glafic.set_secondary('chi2_checknimg 0', verb=0)
            glafic.set_secondary('chi2_restart   -1', verb=0)
            glafic.set_secondary('chi2_usemag    1', verb=0)
            glafic.set_secondary('hvary          0', verb=0)
            glafic.set_secondary('ran_seed -122000', verb=0)
            glafic.startup_setnum(2, 0, 1)
            glafic.set_lens(1, 'sie', 0.261343256161012, 1.30e+02, 20.81, 20.76, 0.107, 23.38, 0.0, 0.0)
            glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.78, 20.78, m[i], n[j], 0.0, 1.0)
            glafic.set_point(1, 1.0, 20.78, 20.78)
            glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
            glafic.setopt_lens(2, 0, 0, 0, 0, 1, 1, 0, 1)
            glafic.setopt_point(1, 0, 1, 1)
            glafic.model_init(verb=0)
            glafic.readobs_point(constraint_file)
            if prior_file:
                glafic.parprior(prior_file)
            glafic.optimize()
            glafic.findimg()
            # glafic.writecrit(1.0)
            # glafic.writelens(1.0)
            glafic.quit()