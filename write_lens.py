#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/NFW+SHEAR/NFW_POS_SHEAR', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, "anfw", 0.261343256161012, 207412400000.0, 0.03195999999999799, -0.026720000000000965, 0.158977, 8.473357, 87.01437, 0.0)
glafic.set_lens(2, "sers", 0.261343256161012, 81418430000.0, 0.03195999999999799, -0.026720000000000965, 0.19174, 97.1557, 51.881, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 9976928000.0, 0.03195999999999799, -0.026720000000000965, 0.58185, 80.9088, 5.97926, 0.5737)
glafic.set_lens(4, "pert", 0.261343256161012, 1.0, 0.045479999999997744, -0.13572000000000273, 0.02684478, 133.0918, 0.0, 0.0)
glafic.set_point(1, 1.000, 0.03862999999999772, -0.0012400000000027944)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()