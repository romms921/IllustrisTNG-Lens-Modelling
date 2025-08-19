# # Sim 1 
# #!/usr/bin/env python
# import glafic

# m = [1, 2, 3, 4, 5]
# n = [1, 2, 3, 4, 5]

# for i in range(len(m)):
#     for j in range(len(n)):

#         glafic.init(0.3, 0.7, -1.0, 0.7, f'MPOLE/POW+MPOLE/POW_POS_MPOLE_{m[i]}_{n[j]}', 20.0, 20.0, 21.56, 21.56, 0.001, 0.001, 1, verb = 0)

#         glafic.set_secondary('chi2_splane 1', verb = 0)
#         glafic.set_secondary('chi2_checknimg 1', verb = 0)
#         glafic.set_secondary('chi2_restart   -1', verb = 0)
#         glafic.set_secondary('chi2_usemag    1', verb = 0)
#         glafic.set_secondary('hvary          0', verb = 0)
#         glafic.set_secondary('ran_seed -122000', verb = 0)

#         glafic.startup_setnum(2, 0, 1)
#         glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 20.78, 20.78, 0.1, -4.0, 0.5, 2.1)
#         glafic.set_lens(2, 'mpole', 0.261343256161012, 1.0, 20.80, 20.75, 0.01, -90.0, m[i], n[j])
#         glafic.set_point(1, 1.0, 20.78, 20.78)

#         glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 1)
#         glafic.setopt_lens(2, 0, 0, 1, 1, 1, 1, 0, 0)
#         glafic.setopt_point(1, 0, 1, 1)

#         # model_init needs to be done again whenever model parameters are changed
#         glafic.model_init(verb = 0)

#         glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
#         glafic.parprior('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/MPOLE/priorfile.dat')
#         glafic.optimize()
#         glafic.findimg()
#         glafic.writecrit(1.0)
#         glafic.writelens(1.0)

#         glafic.quit()

#!/usr/bin/env python
import glafic

m = [1, 2, 3, 4, 5]

for i in range(len(m)):
    glafic.init(0.3, 0.7, -1.0, 0.7, f'MPOLE/POW+MPOLE/POW_POS_MPOLE_{m[i]}', 20.0, 20.0, 21.56, 21.56, 0.001, 0.001, 1, verb = 0)

    glafic.set_secondary('chi2_splane 1', verb = 0)
    glafic.set_secondary('chi2_checknimg 1', verb = 0)
    glafic.set_secondary('chi2_restart   -1', verb = 0)
    glafic.set_secondary('chi2_usemag    1', verb = 0)
    glafic.set_secondary('hvary          0', verb = 0)
    glafic.set_secondary('ran_seed -122000', verb = 0)

    glafic.startup_setnum(2, 0, 1)
    glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 20.78, 20.78, 0.1, -4.0, 0.5, 2.0)
    glafic.set_lens(2, 'mpole', 0.261343256161012, 1.0, 20.80, 20.75, 0.01, -90.0, m[i], 1)
    glafic.set_point(1, 1.0, 20.78, 20.78)

    glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 1)
    glafic.setopt_lens(2, 0, 0, 1, 1, 1, 1, 0, 1)
    glafic.setopt_point(1, 0, 1, 1)

    # model_init needs to be done again whenever model parameters are changed
    glafic.model_init(verb = 0)

    glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
    glafic.parprior('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/MPOLE/priorfile.dat')
    glafic.optimize()
    glafic.findimg()
    glafic.writecrit(1.0)
    glafic.writelens(1.0)

    glafic.quit()