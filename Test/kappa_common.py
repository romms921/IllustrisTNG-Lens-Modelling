#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, "Test/SIE+SHEAR+FIXED/SIE_POS_SHEAR_macro", 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(2, 0, 1)
glafic.set_lens(1, "sie", 0.261343256161012, 158.0026, 20.81371, 20.78384, 0.1862623, 74.13216, 0.0, 0.0)
glafic.set_lens(2, "pert", 0.261343256161012, 1.0, 20.78, 20.78, 0.06086426, -17.08111, 0.0, 0.0)
glafic.set_point(1, 1.000, 20.81165, 20.78101)

glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 0, 0)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
glafic.writelens(1.0)

glafic.quit()