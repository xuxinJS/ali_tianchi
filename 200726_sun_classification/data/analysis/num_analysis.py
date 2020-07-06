import os

alpha = '/home/xuxin/data/sun_classification/images/continuum/alpha'
beta = '/home/xuxin/data/sun_classification/images/continuum/beta'
betax = '/home/xuxin/data/sun_classification/images/continuum/betax'

alpha_target = set()
beta_target = set()
betax_target = set()

for i in os.listdir(alpha):
    target = i.split('_')[0]
    alpha_target.add(target)

for i in os.listdir(beta):
    target = i.split('_')[0]
    beta_target.add(target)

for i in os.listdir(betax):
    target = i.split('_')[0]
    betax_target.add(target)

print("alpha_target:", len(alpha_target), alpha_target)
print("beta_target:", len(beta_target), beta_target)
print("betax_target:", len(betax_target), betax_target)
print("alpha&beta:", len(alpha_target & beta_target), alpha_target & beta_target)
print("alpha&betax:", len(alpha_target & betax_target), alpha_target & betax_target)
print("beta&betax:", len(beta_target & betax_target), beta_target & betax_target)

