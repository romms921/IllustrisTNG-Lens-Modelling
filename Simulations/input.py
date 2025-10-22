import glafic

path = '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/System 2/'
constraint_file = path + 'pos_point.dat'
prior_file = None

glafic.init(0.3, 0.7, -1.0, 0.7, path, -3.0, -3.0, 3.0, 3.0, 0.01, 0.01, 1, verb=0)
glafic.set_secondary('chi2_splane 1', verb=0)
glafic.set_secondary('chi2_checknimg 0', verb=0)
glafic.set_secondary('chi2_restart   -1', verb=0)
glafic.set_secondary('chi2_usemag    1', verb=0)
glafic.set_secondary('hvary          0', verb=0)
glafic.set_secondary('ran_seed -122000', verb=0)
glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, 'sie', 0.261343256161012, 1.563051e+02, 0.0, 0.0, 2.168966e-01, -1.398259e+00,  0.0, 0.0)
glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
glafic.set_point(1, 1.0, 0.0, 0.0)
glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
glafic.setopt_lens(2, 0, 0, 1, 1, 1, 1, 0, 1)
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