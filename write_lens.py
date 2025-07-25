#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Write Lens/POW_POS+FLUX_SHEAR', -0.8176899999999978, -0.8251799999999985, 0.7423100000000022, 0.7348200000000016, 0.0012, 0.0012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, "pow", 0.261343256161012, 1.0, 0, 0, 0.1120995, -5.964733, 0.2080619, 1.777869)
glafic.set_lens(2, "sers", 0.261343256161012, 50047960000.0, 0, 0, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 9463378000.0, 0, 0, 0.58185, 80.9088, 0.2391704, 0.5737)
glafic.set_lens(4, "pert", 0.261343256161012, 1.0, 0, 0, 6.030569e-08, 173.9923, 0.0, 0.0)
glafic.set_point(1, 1.000, 0.033269999999998134, 0.0030199999999993565)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()