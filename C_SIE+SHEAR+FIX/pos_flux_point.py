#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'C_SIE+SHEAR+FIX/SIE_POS+FLUX_SHEAR', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, 'sie', 0.261343256161012, 1.44e+02,  20.81, 20.77, 0.11, 45.93, 0.02, 0.0)
glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.89, 20.60, 0.2, 153.0, 0.0, 0.0)
glafic.set_point(1, 1.0, 20.78, 20.78)

glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 0, 1, 1, 1, 1, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS+FLUX).dat')
glafic.optimize()
glafic.findimg()
glafic.writecrit(1.0)
glafic.writelens(1.0)

glafic.quit()