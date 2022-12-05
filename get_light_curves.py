import pandas as pd
from alerce.core import Alerce
from tqdm import tqdm


alerce_client = Alerce()

sn_ztf = pd.read_parquet('ztf_confirmed_sn.parquet')
oids = sn_ztf['Disc. Internal Name'].values
lcs = []
not_available = []
for oid in tqdm(oids):
    try:
        lc = alerce_client.query_detections(oid, format='pandas')
        lc['oid'] = oid
        lcs.append(lc)
    except:
        not_available.append(oid)


lcs = pd.concat(lcs)
lcs.to_parquet('ztf_sn_lightcurves.parquet')

pd.DataFrame(not_available).to_csv('not_available.csv')
