from generate_spectral import generate_spectral
from STResNet.scripts.papers.AAAI17.BikeNYC.exptBikeNYC import STResNetFunc
from ST3DNet.trainNY import ST3DNetFunc
import time

cluster = 30
iteration = 5
model = 'STResNet'

start_time = time.time()
print('start train:', start_time)

generate_spectral(cluster, iteration)

print('***generate_spectral***')

if model == 'STResNet':
    STResNetFunc(cluster, cluster)
elif model == '3DNet':
    ST3DNetFunc()    
else:
    pass
print('train end:', time.time(), 'time consume: {} s'.format(time.time()-start_time))

