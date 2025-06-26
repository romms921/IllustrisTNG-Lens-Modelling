#!/usr/bin/env python
import glafic

glafic.init(0.3, 0.7, -1.0, 0.7, 'Write Lens/SIE_POS_SHEAR', -0.78, -0.78, 0.78, 0.78, 0.0012, 0.0012, 1, verb = 0)

glafic.set_secondary('chi2_splane 1', verb = 0)
glafic.set_secondary('chi2_checknimg 0', verb = 0)
glafic.set_secondary('chi2_restart   -1', verb = 0)
glafic.set_secondary('chi2_usemag    1', verb = 0)
glafic.set_secondary('hvary          0', verb = 0)
glafic.set_secondary('ran_seed -122000', verb = 0)

glafic.startup_setnum(4, 0, 1)
glafic.set_lens(1, "sie", 0.261343256161012, 115.632, 0.039939999999997866, 0.03228999999999971, 0.1481973, 28.08245, 0.0, 0.0)
glafic.set_lens(2, "sers", 0.261343256161012, 59349720000.0, 0.039939999999997866, 0.03228999999999971, 0.19174, 97.1557, 2.075204, 1.4424)
glafic.set_lens(3, "sers", 0.261343256161012, 7968687000.0, 0.039939999999997866, 0.03228999999999971, 0.58185, 80.9088, 0.2391704, 0.5737)
glafic.set_lens(4, "pert", 0.261343256161012, 1.0, 0.14143999999999934, -0.2092400000000012, 0.02943948, 143.8026, 0.0, 0.0)
glafic.set_point(1, 1.000, 0.04338999999999871, -0.00396000000000285)

glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 0)
glafic.setopt_lens(2, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(3, 0, 1, 1, 1, 0, 0, 0, 0)
glafic.setopt_lens(4, 0, 0, 0, 0, 0, 0, 0, 0)
glafic.setopt_point(1, 0, 1, 1)

# model_init needs to be done again whenever model parameters are changed
glafic.model_init(verb = 0)

glafic.writelens(1.0)

glafic.quit()