#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/POW/POW_POS+FLUX', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(3, 0, 1)
# pow index is 2.1 for POS constraint
glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 2.080976e+01, 2.075982e+01, 1.237922e-01, -4.493337e+00, 4.895203e-01, 1.777869e+00) 
glafic.set_lens(2, 'sers', 0.261343256161012, 7.950299e+10, 21.2807, 20.8498, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, 'sers', 0.261343256161012, 9.913657e+09, 21.2807, 20.8498, 0.58185, 80.9088, 0.2391704, 0.5737)
glafic.set_point(1, 1.0, 2.081362e+01, 2.078027e+01)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 0, 0, 0, 0, 0, 0)
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