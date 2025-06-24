#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Sersic/POW+SHEAR/POW_POS_SHEAR', -0.78, -0.78, 0.78, 0.78, 0.012, 0.012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, "pow", 0.261343256161012, 1.0, 0.029999999999997584, -0.019999999999999574, 0.1608124, 12.24873, 0.4896239, 2.24)
glafic.set_lens(2, "sers", 0.261343256161012, 66918670000.0, 0.029999999999997584, -0.019999999999999574, 0.19174, 97.1557, 51.881, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 9654264000.0, 0.029999999999997584, -0.019999999999999574, 0.58185, 80.9088, 5.97926, 0.5737)
glafic.set_lens(4, "pert", 0.261343256161012, 1.0, 0.030249999999998778, -0.1943400000000004, 0.03085639, 144.5709, 0.0, 0.0)
glafic.set_point(1, 1.000, 0.03863999999999734, 0.006699999999998596)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()