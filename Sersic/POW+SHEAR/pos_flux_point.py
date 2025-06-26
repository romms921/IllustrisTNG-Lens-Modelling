#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/POW+SHEAR/POW_POS+FLUX_SHEAR', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 2.081754e+01, 2.082199e+01, 5.837575e-02, -1.346423e+01, 2.705313e-01, 1.777869e+00)
glafic.set_lens(2, 'sers', 0.261343256161012, 8e+10, 20.8512217, 20.8340052, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, 'sers', 0.261343256161012, 1e+10, 20.8512217, 20.8340052, 0.58185, 80.9088, 0.2391704, 0.5737)
glafic.set_lens(4, 'pert', 0.261343256161012, 1.0, 2.085513e+01, 2.048605e+01, 1.749327e-02, 1.718205e+02, 0.000000e+00, 0.0)
glafic.set_point(1, 1.0, 2.081327e+01, 2.078144e+01)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 1, 1, 1, 1, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS+FLUX).dat')
glafic.parprior("/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/priorfile.dat")
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)

glafic.quit()