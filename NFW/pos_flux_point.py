#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'NFW/NFW_POS+FLUX', 0.0, 0.0, 5.0, 5.0, 0.04, 0.04, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    0', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)
glafic.set_secondary('flag_hodensity 2', verb = 0)

glafic.startup_setnum(1, 0, 1)
glafic.set_lens(1, 'anfw', 0.261343256161012, 1e+12, 2.5, 2.5, 4.0e-01, 0, 30.0, 0.0)
glafic.set_point(1, 1.0, 2.5, 2.5)

glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 1, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS+FLUX).dat')
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)

glafic.quit()