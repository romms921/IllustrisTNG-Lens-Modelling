import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/Simulation/mpole_1_good', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb=0)

glafic.set_secondary('chi2_splane 1', verb=0)
glafic.set_secondary('chi2_checknimg 0', verb=0)
glafic.set_secondary('chi2_restart   -1', verb=0)
glafic.set_secondary('chi2_usemag    1', verb=0)
glafic.set_secondary('hvary          0', verb=0)
glafic.set_secondary('ran_seed -122000', verb=0)
glafic.startup_setnum(2, 0, 1)

glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 20.78, 20.78, 0.107, 23.38, 0.46, 2.1)
glafic.set_lens(2, 'mpole', 0.261343256161012, 1.0, 20.78, 20.78, 0.09861, 188.10811, 1.0, 1.0)
glafic.set_point(1, 1.0, 20.78, 20.78)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 1)
glafic.setopt_lens(2, 0, 0, 1, 1, 0, 0, 0, 1)
glafic.setopt_point(1, 0, 1, 1)

glafic.model_init(verb=0)
glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.parprior('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/MPOLE/priorfile.dat')
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)
glafic.quit()