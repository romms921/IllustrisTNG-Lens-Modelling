#!/usr/bin/env python
import glafic

centre_x = [20.73, 20.74, 20.75, 20.76, 20.77, 20.78, 20.79, 20.80, 20.81, 20.82, 20.83, 20.84, 20.85, 20.86, 20.87, 20.88]

for i in range(len(centre_x)):
    model_name = f'Test/Shear Shift/SIE_POS_SHEAR_{centre_x[i]}'
    glafic.init(0.3, 0.7, -1.0, 0.7, model_name, 20.0, 20.0, 21.56, 21.56, 0.001, 0.001, 1, verb = 0)

    glafic.set_secondary('chi2_splane 1', verb = 0)
    glafic.set_secondary('chi2_checknimg 0', verb = 0)
    glafic.set_secondary('chi2_restart   -1', verb = 0)
    glafic.set_secondary('chi2_usemag    1', verb = 0)
    glafic.set_secondary('hvary          0', verb = 0)
    glafic.set_secondary('ran_seed -122000', verb = 0)

    glafic.startup_setnum(2, 0, 1)
    glafic.set_lens(1, 'sie', 0.261343256161012, 1.580026e+02, 2.081371e+01, 2.078384e+01, 1.862623e-01, 7.413216e+01, 0.000000e+00, 0.000000e+00)
    glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.78, 20.78, 0.01, 0.0, 0.0, 0.0)
    glafic.set_point(1, 1.0, centre_x[i], 20.78)
    glafic.setopt_lens(1, 0, 0, 0, 0, 0, 0, 0, 0)
    glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
    glafic.setopt_point(1, 0, 0, 0)

    # model_init needs to be done again whenever model parameters are changed
    glafic.model_init(verb = 0)

    glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
    glafic.optimize()
    glafic.findimg()
    glafic.writecrit(1.0)
    glafic.writelens(1.0)

    glafic.quit()