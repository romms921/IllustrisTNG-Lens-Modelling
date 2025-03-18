#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/NFW+SHEAR/NFW_POS_SHEAR', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 0', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)
glafic.set_secondary('flag_hodensity 2', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, 'anfw', 0.261343256161012, 1.841836e+11, 2.081199e+01, 2.075357e+01, 1.579358e-01, 8.657437e+00, 8.701885e+01, 0.000000e+00)
glafic.set_lens(2, 'sers', 0.261343256161012, 8e+10, 21.2807, 20.8498, 0.19174, 97.1557, 51.8810, 1.4424)
glafic.set_lens(3, 'sers', 0.261343256161012, 1e+10, 21.2807, 20.8498, 0.58185, 80.9088, 5.97926, 0.5737)
glafic.set_lens(4, 'pert', 0.261343256161012, 1.0, 2.080820e+01, 2.064165e+01, 2.469017e-02, 1.333287e+02, 0.000000e+00, 0.0)
glafic.set_point(1, 1.0, 2.081794e+01, 2.076874e+01)

glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 1, 1, 1, 1, 0, 1)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)

glafic.quit()