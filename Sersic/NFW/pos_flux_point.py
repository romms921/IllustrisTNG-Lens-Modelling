#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/NFW/NFW_POS+FLUX', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)
glafic.set_secondary('flag_hodensity 2', verb = 0)

glafic.startup_setnum(3, 0, 1)
glafic.set_lens(1, 'anfw', 0.261343256161012, 2.134707e+11, 2.080890e+01, 2.075429e+01, 1.457899e-01, -4.427899e+00, 8.499782e+01, 0.000000e+00)
glafic.set_lens(2, 'sers', 0.261343256161012, 8e+10, 20.8512217, 20.8340052, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, 'sers', 0.261343256161012, 1e+10, 20.8512217, 20.8340052, 0.58185, 80.9088, 0.2391704, 0.5737)
glafic.set_point(1, 1.0, 2.081265e+01, 2.077988e+01)

glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
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